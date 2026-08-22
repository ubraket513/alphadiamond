#include "native_controller.hpp"

#include <QDir>
#include <QCoreApplication>
#include <QDebug>
#include <QDateTime>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointF>
#include <QSet>
#include <QTimer>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <numeric>
#include <map>
#include <tuple>
#include <vector>

#include "soo/board.hpp"
#include "soo/rules.hpp"
#include "native_move_player.hpp"

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_evaluator.hpp"
#include "soo/mcts.hpp"

namespace {
void configure_torch_cpu() {
    static std::once_flag configured;
    std::call_once(configured, [] {
        bool ok = false;
        const int requested = qEnvironmentVariableIntValue("DIAMOND_TORCH_THREADS", &ok);
        const int threads = ok && requested > 0 ? requested : 1;
        torch::set_num_threads(threads);
        torch::set_num_interop_threads(1);
    });
}
}
#endif

namespace {
struct BoardPoint { int x; int y; int z; };

std::vector<BoardPoint> board_points() {
    std::vector<BoardPoint> points;
    for (int x = -13; x <= 13; ++x) for (int z = -13; z <= 13; ++z) {
        const int y = -x - z;
        const bool up = x >= -3 && y >= -3 && z >= -3;
        const bool down = x <= 3 && y <= 3 && z <= 3;
        if (up || down) points.push_back({x, y, z});
    }
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
        return a.z == b.z ? a.x < b.x : a.z < b.z;
    });
    return points;
}

QPointF unit_xy(const BoardPoint& point) {
    return {point.x + point.z / 2.0, point.z * std::sqrt(3.0) / 2.0};
}

bool point_in_camp(const BoardPoint& point, int camp) {
    const int values[3] = {point.x, point.y, point.z};
    const bool triangle_up = point.x >= -3 && point.y >= -3 && point.z >= -3;
    const bool triangle_down = point.x <= 3 && point.y <= 3 && point.z <= 3;
    return camp < 3 ? values[camp] >= 3 && triangle_up
                    : values[camp - 3] <= -3 && triangle_down;
}

QString camp_key(int camp) {
    static const QStringList keys = {QStringLiteral("x+"), QStringLiteral("y+"),
        QStringLiteral("z+"), QStringLiteral("x-"), QStringLiteral("y-"),
        QStringLiteral("z-")};
    return camp >= 0 && camp < keys.size() ? keys[camp] : QString();
}
}

GeometryModel::GeometryModel(QObject* parent) : QObject(parent) {}

void GeometryModel::setPlayerCount(int count) {
    player_count_ = count;
    Q_EMIT changed();
}

QVariantMap GeometryModel::bounds() const {
    const auto points = board_points();
    double min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;
    for (const auto& point : points) {
        const auto xy = unit_xy(point);
        min_x = std::min(min_x, xy.x()); min_y = std::min(min_y, xy.y());
        max_x = std::max(max_x, xy.x()); max_y = std::max(max_y, xy.y());
    }
    return {{"minX", min_x}, {"minY", min_y}, {"maxX", max_x}, {"maxY", max_y}};
}

QVariantList GeometryModel::holes() const {
    QVariantList result;
    const auto points = board_points();
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const auto xy = unit_xy(points[i]);
        result.push_back(QVariantMap{{"id", i}, {"x", xy.x()}, {"y", xy.y()}});
    }
    return result;
}

QVariantList GeometryModel::edges() const {
    QVariantList result;
    const auto points = board_points();
    std::map<std::tuple<int, int, int>, int> ids;
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        ids[{points[i].x, points[i].y, points[i].z}] = i;
    const int directions[6][3] = {{1,-1,0},{1,0,-1},{0,1,-1},{-1,1,0},{-1,0,1},{0,-1,1}};
    for (int i = 0; i < static_cast<int>(points.size()); ++i) for (const auto& d : directions) {
        const auto key = std::make_tuple(points[i].x + d[0], points[i].y + d[1], points[i].z + d[2]);
        const auto it = ids.find(key);
        if (it == ids.end() || i >= it->second) continue;
        const auto a = unit_xy(points[i]); const auto b = unit_xy(points[it->second]);
        result.push_back(QVariantMap{{"x1", a.x()}, {"y1", a.y()}, {"x2", b.x()}, {"y2", b.y()}});
    }
    return result;
}

QVariantList GeometryModel::camps() const {
    QVariantList result;
    const auto points = board_points();
    for (int camp = 0; camp < 6; ++camp) {
        std::vector<std::pair<int, QPointF>> camp_candidates;
        for (int position = 0; position < static_cast<int>(points.size()); ++position) {
            const auto& point = points[position];
            if (!point_in_camp(point, camp)) continue;
            const auto xy = unit_xy(point);
            camp_candidates.push_back({position, xy});
        }
        double centre_x = 0.0, centre_y = 0.0;
        for (const auto& candidate : camp_candidates) { centre_x += candidate.second.x(); centre_y += candidate.second.y(); }
        if (!camp_candidates.empty()) { centre_x /= camp_candidates.size(); centre_y /= camp_candidates.size(); }
        std::sort(camp_candidates.begin(), camp_candidates.end(), [centre_x, centre_y](const auto& a, const auto& b) {
            const auto da = std::pow(a.second.x() - centre_x, 2) + std::pow(a.second.y() - centre_y, 2);
            const auto db = std::pow(b.second.x() - centre_x, 2) + std::pow(b.second.y() - centre_y, 2);
            return da == db ? a.first < b.first : da > db;
        });
        QVariantList camp_points;
        for (int i = 0; i < std::min(3, static_cast<int>(camp_candidates.size())); ++i)
            camp_points.push_back(QVariantMap{{"x", camp_candidates[i].second.x()}, {"y", camp_candidates[i].second.y()}});
        QVariantList camp_holes;
        for (const auto& candidate : camp_candidates)
            camp_holes.push_back(QVariantMap{{"x", candidate.second.x()}, {"y", candidate.second.y()}});
        result.push_back(QVariantMap{
            {"key", camp_key(camp)},
            {"inPlay", camp != 1 && camp != 4 || player_count_ == 3},
            {"color", camp == 0 || camp == 3 ? "#34C759" : camp == 1 || camp == 4 ? "#FFCC00" : "#FF3B30"},
            {"points", camp_points}, {"holes", camp_holes}});
    }
    return result;
}

ContractListModel::ContractListModel(QString kind, QObject* parent)
    : QAbstractListModel(parent), kind_(std::move(kind)) {
    const QList<QPair<int, QByteArray>> common = {
        {PositionIdRole, "positionId"}, {UnitXRole, "unitX"}, {UnitYRole, "unitY"},
        {ColorRole, "color"}, {PlayerIdRole, "playerId"}};
    for (const auto& role : common) roles_.insert(role.first, role.second);
    if (kind_ == "board") {
        roles_.insert(CampKeyRole, "campKey"); roles_.insert(OccupantRole, "occupant");
        roles_.insert(IsSelectedRole, "isSelected"); roles_.insert(IsLegalStepRole, "isLegalStep");
        roles_.insert(IsLegalJumpRole, "isLegalJump"); roles_.insert(IsPathNodeRole, "isPathNode");
        roles_.insert(PathIndexRole, "pathIndex"); roles_.insert(IsLastMoveSourceRole, "isLastMoveSource");
        roles_.insert(IsLastMoveDestRole, "isLastMoveDest"); roles_.insert(IsProposalSourceRole, "isProposalSource");
        roles_.insert(IsProposalDestRole, "isProposalDest");
    } else if (kind_ == "piece") {
        roles_.insert(PieceIdRole, "pieceId"); roles_.insert(IsMovingRole, "isMoving");
    } else if (kind_ == "history") {
        roles_.insert(TurnNumberRole, "turnNumber"); roles_.insert(PlayerLabelRole, "playerLabel");
        roles_.insert(PlayerColorRole, "playerColor"); roles_.insert(MoveTextRole, "moveText");
        roles_.insert(PathTextRole, "pathText"); roles_.insert(HopCountRole, "hopCount");
        roles_.insert(IsAiRole, "isAi");
    } else if (kind_ == "players") {
        roles_.insert(NameRole, "name"); roles_.insert(KindLabelRole, "kindLabel");
        roles_.insert(IsCurrentRole, "isCurrent"); roles_.insert(IsAiRole, "isAi");
        roles_.insert(HomeCountRole, "homeCount"); roles_.insert(CampSizeRole, "campSize");
        roles_.insert(HasFinishedRole, "hasFinished"); roles_.insert(TurnIndexRole, "turnIndex");
        roles_.insert(PlaceRole, "place"); roles_.insert(PlaceLabelRole, "placeLabel");
    }
}

int ContractListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return rows_.size();
}

QVariant ContractListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    const QByteArray role_name = roles_.value(role);
    return role_name.isEmpty() ? QVariant{} : index.row() < rows_.size()
        ? rows_.at(index.row()).toMap().value(QString::fromUtf8(role_name)) : QVariant{};
}

QHash<int, QByteArray> ContractListModel::roleNames() const {
    return roles_;
}

void ContractListModel::setRows(QVariantList rows) {
    beginResetModel();
    rows_ = std::move(rows);
    endResetModel();
}

int ContractListModel::rowWithValue(const QByteArray& roleName, const QVariant& value) const {
    const QString key = QString::fromUtf8(roleName);
    for (int row = 0; row < rows_.size(); ++row) {
        if (rows_.at(row).toMap().value(key) == value) return row;
    }
    return -1;
}

void ContractListModel::updateRow(int row, const QVariantMap& values) {
    if (row < 0 || row >= rows_.size()) return;
    QVariantMap current = rows_.at(row).toMap();
    for (auto it = values.cbegin(); it != values.cend(); ++it) current.insert(it.key(), it.value());
    rows_[row] = current;
    const QModelIndex changed_index = index(row, 0);
    Q_EMIT dataChanged(changed_index, changed_index);
}

NativeController::NativeController(QObject* parent) : QObject(parent) {
    bool simulations_ok = false;
    const int configured_simulations = qEnvironmentVariableIntValue(
        "DIAMOND_MCTS_SIMULATIONS", &simulations_ok);
    if (simulations_ok && configured_simulations > 0 && configured_simulations <= 4096)
        ai_simulations_ = configured_simulations;
    geometry_ = new GeometryModel(this);
    board_model_ = new ContractListModel("board", this);
    piece_model_ = new ContractListModel("piece", this);
    history_model_ = new ContractListModel("history", this);
    player_model_ = new ContractListModel("players", this);
    ai_worker_ = new NativeAiWorker(this);
    sound_player_ = new NativeMovePlayer(this);
    connect(sound_player_, &NativeMovePlayer::changed, this, &NativeController::changed);
    animation_timer_ = new QTimer(this);
    animation_timer_->setInterval(140);
    connect(animation_timer_, &QTimer::timeout, this, &NativeController::animationTick);
    connect(ai_worker_, &NativeAiWorker::resultReady, this,
            [this](quint64 generation, int action) {
                if (generation != generation_ || !ai_thinking_) return;
                std::vector<int32_t> legal;
                soo::legal_action_ids(state_, legal);
                if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
                    ai_thinking_ = false;
                    ai_failure_latched_ = true;
                    ai_restart_when_idle_ = false;
                    ai_status_ = QStringLiteral("Invalid proposal");
                    qWarning() << "native AI returned an illegal action" << action;
                    fail(QStringLiteral("Agent proposed an illegal move."));
                    return;
                }
                ai_thinking_ = false;
                ai_failure_latched_ = false;
                selected_position_ = -1;
                legal_actions_.clear();
                ai_status_ = QStringLiteral("Proposal ready");
                if (ai_started_at_ms_ > 0) {
                    ai_details_.push_back(QVariantMap{{"label", QStringLiteral("Search time (ms)")},
                        {"value", QString::number(QDateTime::currentMSecsSinceEpoch() - ai_started_at_ms_)}});
                }
                proposeAction(action, true);
            });
    connect(ai_worker_, &NativeAiWorker::failed, this,
            [this](quint64 generation, const QString& message) {
                if (generation != generation_) return;
                ai_thinking_ = false;
                ai_failure_latched_ = true;
                ai_restart_when_idle_ = false;
                ai_status_ = QStringLiteral("Error");
                qWarning() << "native AI worker:" << message;
                fail(QStringLiteral("Agent failed: %1").arg(message));
            });
    connect(ai_worker_, &NativeAiWorker::cancelled, this,
            [this](quint64 generation) {
                if (generation == generation_) {
                    ai_thinking_ = false;
                    ai_status_ = QStringLiteral("Ready");
                    Q_EMIT changed();
                }
            });
    connect(ai_worker_, &NativeAiWorker::becameIdle, this, [this] {
        if (!ai_restart_when_idle_) return;
        ai_restart_when_idle_ = false;
        if (!ai_failure_latched_ && !ai_thinking_ && proposal_action_ < 0 &&
            !isGameOver() && isCurrentPlayerAi())
            QTimer::singleShot(0, this, &NativeController::startAiTurn);
    });
    match_.count = 2;
    match_.players[0] = soo::PlayerSpec{1, 2, 5};
    match_.players[1] = soo::PlayerSpec{2, 0, 3};
    ai_seats_ = {2};
    geometry_->setPlayerCount(match_.count);
    state_.current_player = 1;
    loadTopology();
    if (soo::mutable_topology().configured) {
        for (int camp = 0; camp < match_.count; ++camp) {
            const auto& spec = match_.players[camp];
            for (uint8_t position : soo::topology().camp_positions[spec.camp])
                state_.occupancy[position] = spec.id;
        }
    }
    refreshModels();
}

NativeController::~NativeController() { shutdown(); }

void NativeController::cancelSearch() {
    ++generation_;
    ai_restart_when_idle_ = false;
    if (ai_worker_) ai_worker_->cancel();
    ai_thinking_ = false;
}

QString NativeController::aiAgentName() const {
#ifdef DIAMOND_QT_HAS_SOO
    return match_.count == 2 ? QStringLiteral("Soo AlphaZero")
                             : QStringLiteral("Native fallback");
#else
    return QStringLiteral("Native deterministic");
#endif
}

QString NativeController::phase() const {
    if (isGameOver()) return QStringLiteral("GAME_OVER");
    if (animating_) return QStringLiteral("ANIMATING_MOVE");
    if (ai_thinking_) return QStringLiteral("AI_THINKING");
    if (proposal_is_ai_) return QStringLiteral("AI_MOVE_PROPOSED");
    if (proposal_action_ >= 0) return QStringLiteral("HUMAN_MOVE_PROPOSED");
    return QStringLiteral("WAITING_FOR_HUMAN_INPUT");
}

QUrl NativeController::defaultSaveDir() const {
    return QUrl::fromLocalFile(QDir::homePath() + QStringLiteral("/.alphadiamond/saves"));
}

bool NativeController::soundAvailable() const { return sound_player_->available(); }
bool NativeController::soundEnabled() const { return sound_player_->enabled(); }
QString NativeController::soundStatus() const { return sound_player_->status(); }
double NativeController::soundVolume() const { return sound_player_->volume(); }
bool NativeController::soundLoaded() const { return sound_player_->loaded(); }
int NativeController::soundPlayRequestCount() const { return sound_player_->playRequestCount(); }

void NativeController::previewSound() { sound_player_->play(); }
void NativeController::setSoundEnabled(bool enabled) { sound_player_->setMuted(!enabled); }
void NativeController::setSoundVolume(double volume) { sound_player_->setVolume(volume); }

QVariantList NativeController::standings() const {
    QVariantList result;
    for (int index = 0; index < state_.finished_count; ++index) {
        const auto id = state_.finish_order[index];
        const int place = index + 1;
        const QString place_label = place == 1 ? QStringLiteral("1st")
            : place == 2 ? QStringLiteral("2nd") : place == 3 ? QStringLiteral("3rd") : QString();
        result.push_back(QVariantMap{{"place", place}, {"placeLabel", place_label}, {"playerId", id},
            {"name", playerName(id)}, {"color", playerColor(id)},
            {"isAi", ai_seats_.contains(id)}});
    }
    return result;
}

QString NativeController::resultSummary() const {
    if (!state_.finished_count) return {};
    if (!isGameOver())
        return QStringLiteral("%1 is home in 1st place. Play continues for 2nd.")
            .arg(playerName(state_.finish_order[0]));
    QStringList parts;
    for (int index = 0; index < state_.finished_count; ++index) {
        const QString place = index == 0 ? QStringLiteral("1st")
            : index == 1 ? QStringLiteral("2nd") : QStringLiteral("3rd");
        parts.push_back(QStringLiteral("%1 %2").arg(place, playerName(state_.finish_order[index])));
    }
    return parts.join(QStringLiteral(" · "));
}

QVariantList NativeController::seatColorsFor(int count) const {
    if (count == 2) return {"#FF3B30", "#34C759"};
    return {"#FF3B30", "#FFCC00", "#34C759"};
}

QString NativeController::playerColor(uint8_t id) const {
    if (id == 1) return QStringLiteral("#FF3B30");
    if (id == 2 && match_.count == 3) return QStringLiteral("#FFCC00");
    return QStringLiteral("#34C759");
}

QString NativeController::playerName(uint8_t id) const { return QStringLiteral("Player %1").arg(id); }

QString NativeController::currentPlayerName() const { return playerName(state_.current_player); }
QString NativeController::currentPlayerColor() const { return playerColor(state_.current_player); }

QVariantList NativeController::turnOrder() const {
    QVariantList result;
    for (const auto& player : match_.players) if (player.id != 0) result.push_back(player.id);
    return result;
}

QVariantList NativeController::aiSeats() const { return ai_seats_; }

QVariantList NativeController::proposalPathIds() const {
    QVariantList result;
    for (int position : proposal_path_) result.push_back(position);
    return result;
}

QString NativeController::proposalSummary() const {
    if (proposal_path_.size() < 2) return {};
    return QStringLiteral("%1 → %2").arg(proposal_path_.front()).arg(proposal_path_.back());
}

QString NativeController::proposalPath() const {
    QStringList positions;
    for (int position : proposal_path_) positions.push_back(QString::number(position));
    return positions.join(QStringLiteral(" → "));
}

QString NativeController::lastMoveText() const {
    if (history_.isEmpty()) return QStringLiteral("—");
    const QVariantMap row = history_.constLast().toMap();
    return QStringLiteral("%1  %2").arg(row.value("playerLabel").toString(),
                                        row.value("moveText").toString());
}

bool NativeController::startMatch(const QVariantList& order, const QVariantList& aiSeats) {
    if (order.size() < 2 || order.size() > 3) {
        fail(QStringLiteral("Cannot start match: unsupported player count."));
        return false;
    }
    QSet<int> seat_ids;
    for (const QVariant& value : order) seat_ids.insert(value.toInt());
    for (int id = 1; id <= order.size(); ++id) {
        if (!seat_ids.contains(id)) {
            fail(QStringLiteral("Cannot start match: turn order must be a seat permutation."));
            return false;
        }
    }
    if (seat_ids.size() != order.size()) {
        fail(QStringLiteral("Cannot start match: duplicate seat in turn order."));
        return false;
    }
    for (const QVariant& value : aiSeats) {
        if (!seat_ids.contains(value.toInt())) {
            fail(QStringLiteral("Cannot start match: AI seat is not in this match."));
            return false;
        }
    }
    cancelSearch();
    ai_failure_latched_ = false;
    match_ = {};
    match_.count = static_cast<uint8_t>(order.size());
    geometry_->setPlayerCount(match_.count);
    const int camps2[2] = {2, 0};
    const int targets2[2] = {5, 3};
    const int camps3[3] = {2, 1, 0};
    const int targets3[3] = {5, 4, 3};
    for (int i = 0; i < match_.count; ++i) {
        const int id = order.at(i).toInt();
        const int camp = match_.count == 2 ? camps2[id - 1] : camps3[id - 1];
        const int target = match_.count == 2 ? targets2[id - 1] : targets3[id - 1];
        match_.players[i] = soo::PlayerSpec{static_cast<uint8_t>(id),
                                             static_cast<uint8_t>(camp),
                                             static_cast<uint8_t>(target)};
    }
    ai_seats_ = aiSeats;
    state_ = {};
    state_.current_player = static_cast<uint8_t>(order.at(0).toInt());
    stopAnimation();
    state_history_.clear(); history_.clear(); selected_position_ = -1; legal_actions_.clear();
    clearProposal();
    piece_model_->setRows({});
    next_piece_id_ = 0;
    ai_rejected_.clear();
    ai_details_.clear();
    ai_status_ = QStringLiteral("Ready");
    announced_finishers_.clear();
    last_action_ = -1;
    ++game_number_;
    error_message_.clear();
    status_message_ = QStringLiteral("New %1-player match started.").arg(match_.count);
    for (int i = 0; i < match_.count; ++i)
        for (const auto position : soo::topology().camp_positions[match_.players[i].camp])
            state_.occupancy[position] = match_.players[i].id;
    refreshModels(); Q_EMIT changed();
    if (ai_seats_.contains(state_.current_player)) startAiTurn();
    return true;
}

void NativeController::newGame() {
    const QVariantList order = turnOrder();
    const QVariantList ai = aiSeats();
    startMatch(order, ai);
    status_message_ = QStringLiteral("New game started.");
    Q_EMIT changed();
}

bool NativeController::saveGame(const QUrl& path) {
    if (!path.isLocalFile()) {
        fail(QStringLiteral("Save failed: destination is not a local file."));
        return false;
    }
    QJsonObject root;
    root["schema_version"] = 2;
    QJsonArray players;
    for (int index = 0; index < match_.count; ++index) {
        const auto& player = match_.players[index];
        players.push_back(QJsonObject{{"id", player.id}, {"name", playerName(player.id)},
            {"kind", ai_seats_.contains(player.id) ? QStringLiteral("ai") : QStringLiteral("human")},
            {"camp", camp_key(player.camp)}, {"target_camp", camp_key(player.target_camp)},
            {"color", playerColor(player.id)}});
    }
    root["players"] = players;
    QJsonArray occupancy;
    for (const auto piece : state_.occupancy) occupancy.push_back(piece);
    root["occupancy"] = occupancy;
    root["current_player_id"] = state_.current_player;
    root["turn_number"] = state_.turn_number;
    root["status"] = isGameOver() ? QStringLiteral("finished") : QStringLiteral("in_progress");
    QJsonArray finish_order;
    for (int index = 0; index < state_.finished_count; ++index)
        finish_order.push_back(state_.finish_order[index]);
    root["finish_order"] = finish_order;

    QJsonArray records;
    for (const QVariant& value : history_) {
        const QVariantMap row = value.toMap();
        QJsonArray move_path;
        for (const QVariant& position : row.value("pathIds").toList())
            move_path.push_back(position.toInt());
        QJsonObject move{{"player_id", row.value("playerId").toInt()},
            {"source", row.value("source").toInt()},
            {"destination", row.value("destination").toInt()}, {"path", move_path},
            {"kind", row.value("kind").toString()}};
        records.push_back(QJsonObject{{"turn_number", row.value("turnNumber").toInt()},
            {"player_id", row.value("playerId").toInt()}, {"move", move},
            {"timestamp", QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
            {"metadata", QJsonObject{}}});
    }
    root["history"] = records;

    const QString local_path = path.toLocalFile();
    QDir().mkpath(QFileInfo(local_path).absolutePath());
    QFile file(local_path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        fail(QStringLiteral("Save failed: %1").arg(file.errorString()));
        return false;
    }
    if (file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0) {
        fail(QStringLiteral("Save failed: %1").arg(file.errorString()));
        return false;
    }
    status_message_ = QStringLiteral("Saved to %1").arg(local_path);
    error_message_.clear();
    Q_EMIT changed();
    return true;
}

bool NativeController::loadGame(const QUrl& path) {
    if (!path.isLocalFile()) {
        fail(QStringLiteral("Load failed: source is not a local file."));
        return false;
    }
    QFile file(path.toLocalFile());
    if (!file.open(QIODevice::ReadOnly)) {
        fail(QStringLiteral("Load failed: %1").arg(file.errorString()));
        return false;
    }
    QJsonParseError parse_error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (parse_error.error != QJsonParseError::NoError || !document.isObject()) {
        fail(QStringLiteral("Load failed: invalid JSON (%1).").arg(parse_error.errorString()));
        return false;
    }
    const QJsonObject root = document.object();
    QVariantList order, ai;
    const int schema = root.value("schema_version").toInt(0);
    if (schema == 2) {
        for (const QJsonValue& value : root.value("players").toArray()) {
            const QJsonObject player = value.toObject();
            order.push_back(player.value("id").toInt());
            if (player.value("kind").toString() == QStringLiteral("ai")) ai.push_back(player.value("id").toInt());
        }
    } else if (root.value("version").toInt() == 1) {
        for (const QJsonValue& value : root.value("order").toArray()) order.push_back(value.toInt());
        for (const QJsonValue& value : root.value("aiSeats").toArray()) ai.push_back(value.toInt());
    } else {
        fail(QStringLiteral("Load failed: unsupported save schema version."));
        return false;
    }
    if (!startMatch(order, ai)) return false;
    cancelSearch();

    const auto occupancy = root.value("occupancy").toArray();
    if (occupancy.size() != soo::kBoardSize) {
        fail(QStringLiteral("Load failed: occupancy must contain 73 holes."));
        return false;
    }

    if (schema == 2) {
        history_.clear();
        state_history_.clear();
        for (const QJsonValue& value : root.value("history").toArray()) {
            const QJsonObject entry = value.toObject();
            const QJsonObject move = entry.value("move").toObject();
            const int source = move.value("source").toInt(-1);
            const int destination = move.value("destination").toInt(-1);
            const int32_t action = source * soo::kBoardSize + destination;
            std::vector<int32_t> legal;
            soo::legal_action_ids(state_, legal);
            if (move.value("player_id").toInt() != state_.current_player ||
                std::find(legal.begin(), legal.end(), action) == legal.end()) {
                fail(QStringLiteral("Load failed: saved history contains an illegal move."));
                return false;
            }
            std::vector<uint8_t> canonical;
            uint8_t kind = soo::kStep;
            soo::canonical_move_path(state_, source, destination, canonical, &kind);
            QVariantList path_ids;
            for (uint8_t position : canonical) path_ids.push_back(position);
            const uint8_t player = state_.current_player;
            state_history_.push_back(state_);
            state_ = soo::apply_action(state_, match_, action);
            const bool is_ai = ai_seats_.contains(player);
            QStringList path_parts;
            for (uint8_t position : canonical) path_parts.push_back(QString::number(position));
            history_.push_back(QVariantMap{{"turnNumber", entry.value("turn_number").toInt()},
                {"playerId", player}, {"playerLabel", is_ai ? QStringLiteral("AI")
                                                              : QStringLiteral("P%1").arg(player)},
                {"playerColor", playerColor(player)}, {"isAi", is_ai},
                {"moveText", QStringLiteral("%1 → %2").arg(source).arg(destination)},
                {"pathText", path_parts.join(QStringLiteral(" → "))}, {"pathIds", path_ids},
                {"source", source}, {"destination", destination},
                {"kind", kind == soo::kJump ? QStringLiteral("jump") : QStringLiteral("step")},
                {"hopCount", std::max(1, static_cast<int>(canonical.size()) - 1)}});
            last_action_ = action;
        }
        for (int index = 0; index < soo::kBoardSize; ++index) {
            if (state_.occupancy[index] != occupancy.at(index).toInt()) {
                fail(QStringLiteral("Load failed: saved board does not match move history."));
                return false;
            }
        }
        if (state_.current_player != root.value("current_player_id").toInt() ||
            state_.turn_number != root.value("turn_number").toInt()) {
            fail(QStringLiteral("Load failed: saved turn does not match move history."));
            return false;
        }
    } else {
        for (int index = 0; index < soo::kBoardSize; ++index)
            state_.occupancy[index] = static_cast<uint8_t>(occupancy.at(index).toInt());
        state_.current_player = static_cast<uint8_t>(root.value("currentPlayer").toInt(state_.current_player));
        state_.turn_number = static_cast<uint16_t>(root.value("turnNumber").toInt(1));
    }
    selected_position_ = -1;
    legal_actions_.clear();
    clearProposal();
    announced_finishers_.clear();
    for (int index = 0; index < state_.finished_count; ++index)
        announced_finishers_.push_back(state_.finish_order[index]);
    status_message_ = QStringLiteral("Game loaded.");
    error_message_.clear();
    refreshModels();
    Q_EMIT changed();
    if (!isGameOver() && ai_seats_.contains(state_.current_player)) startAiTurn();
    return true;
}

void NativeController::loadTopology() {
    QString root = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("artifacts/soo-spike"));
    if (!QDir(root).exists()) root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
    auto& topo = soo::mutable_topology();
    {
        std::ifstream file(root.toStdString() + "/topology_neighbour.i8", std::ios::binary);
        if (!file) return;
        file.read(reinterpret_cast<char*>(topo.neighbour.data()), sizeof(topo.neighbour));
        if (file.gcount() != static_cast<std::streamsize>(sizeof(topo.neighbour))) return;
    }
    std::vector<int32_t> camps(60), pairwise(5329), physical(438), canonical(438);
    auto read_vector = [&](const char* name, std::vector<int32_t>& values) {
        std::ifstream file(root.toStdString() + "/" + name, std::ios::binary);
        if (!file) return false;
        file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(int32_t));
        return file.gcount() == static_cast<std::streamsize>(values.size() * sizeof(int32_t));
    };
    if (!read_vector("topology_camp_positions.i32", camps) ||
        !read_vector("topology_pairwise_distance.i32", pairwise) ||
        !read_vector("topology_physical_to_canonical.i32", physical) ||
        !read_vector("topology_canonical_to_physical.i32", canonical)) return;
    for (int c = 0; c < 6; ++c) for (int p = 0; p < 10; ++p)
        topo.camp_positions[c][p] = static_cast<uint8_t>(camps[c * 10 + p]);
    for (int r = 0; r < 73; ++r) for (int c = 0; c < 73; ++c)
        topo.pairwise[r][c] = static_cast<uint8_t>(pairwise[r * 73 + c]);
    for (int c = 0; c < 6; ++c) for (int p = 0; p < 73; ++p) {
        topo.physical_to_canonical[c][p] = static_cast<uint8_t>(physical[c * 73 + p]);
        topo.canonical_to_physical[c][p] = static_cast<uint8_t>(canonical[c * 73 + p]);
    }
    topo.configured = true;
}

void NativeController::rebuildPieceModel() {
    const auto geo = geometry_->holes();
    QVariantList rows = piece_model_->rows();
    QSet<int> occupied_positions;
    for (int position = 0; position < soo::kBoardSize; ++position) {
        if (state_.occupancy[position] != soo::kEmpty) occupied_positions.insert(position);
    }

    QSet<int> consumed;
    for (int row = 0; row < rows.size(); ++row) {
        QVariantMap piece = rows[row].toMap();
        const int position = piece.value("positionId").toInt();
        const int player = piece.value("playerId").toInt();
        if (position >= 0 && position < soo::kBoardSize && state_.occupancy[position] == player) {
            piece["isMoving"] = false;
            rows[row] = piece;
            consumed.insert(position);
            next_piece_id_ = std::max(next_piece_id_, piece.value("pieceId").toInt() + 1);
        }
    }

    for (int row = 0; row < rows.size(); ++row) {
        QVariantMap piece = rows[row].toMap();
        const int old_position = piece.value("positionId").toInt();
        const int player = piece.value("playerId").toInt();
        if (consumed.contains(old_position) && state_.occupancy[old_position] == player) continue;
        int target = -1;
        for (int position = 0; position < soo::kBoardSize; ++position) {
            if (!consumed.contains(position) && state_.occupancy[position] == player) {
                target = position;
                break;
            }
        }
        if (target < 0) {
            piece["positionId"] = -1;
            rows[row] = piece;
            continue;
        }
        const QVariantMap point = geo[target].toMap();
        piece["positionId"] = target;
        piece["unitX"] = point.value("x");
        piece["unitY"] = point.value("y");
        piece["color"] = playerColor(static_cast<uint8_t>(player));
        piece["isMoving"] = false;
        rows[row] = piece;
        consumed.insert(target);
    }

    QVariantList compact;
    for (const QVariant& value : rows) {
        if (value.toMap().value("positionId").toInt() >= 0) compact.push_back(value);
    }
    for (int position = 0; position < soo::kBoardSize; ++position) {
        if (!occupied_positions.contains(position) || consumed.contains(position)) continue;
        const int player = state_.occupancy[position];
        const QVariantMap point = geo[position].toMap();
        compact.push_back(QVariantMap{{"pieceId", next_piece_id_++}, {"playerId", player},
            {"positionId", position}, {"unitX", point.value("x")},
            {"unitY", point.value("y")}, {"color", playerColor(static_cast<uint8_t>(player))},
            {"isMoving", false}});
    }
    piece_model_->setRows(compact);
}

void NativeController::refreshModels(bool rebuildPieces) {
    QVariantList board;
    const auto geo = geometry_->holes();
    const auto points = board_points();
    const int last_source = last_action_ >= 0 ? last_action_ / soo::kBoardSize : -1;
    const int last_destination = last_action_ >= 0 ? last_action_ % soo::kBoardSize : -1;
    for (int p = 0; p < soo::kBoardSize; ++p) {
        QString key;
        for (int camp = 0; camp < 6; ++camp) {
            if (point_in_camp(points[p], camp)) { key = camp_key(camp); break; }
        }
        QVariantMap row{{"positionId", p}, {"unitX", geo[p].toMap().value("x")},
                        {"unitY", geo[p].toMap().value("y")}, {"campKey", key},
                        {"occupant", state_.occupancy[p]}, {"isSelected", p == selected_position_},
                        {"isLegalStep", false}, {"isLegalJump", false}, {"isPathNode", false},
                        {"pathIndex", -1}, {"isLastMoveSource", p == last_source},
                        {"isLastMoveDest", p == last_destination},
                        {"isProposalSource", false}, {"isProposalDest", false}};
        for (int32_t action : legal_actions_) if (action % soo::kBoardSize == p) {
            std::vector<uint8_t> path;
            uint8_t kind = soo::kStep;
            if (soo::canonical_move_path(state_, action / soo::kBoardSize, p, path, &kind))
                row[kind == soo::kJump ? "isLegalJump" : "isLegalStep"] = true;
        }
        for (int index = 0; index < proposal_path_.size(); ++index) {
            if (proposal_path_[index] != p) continue;
            row["isPathNode"] = true;
            row["pathIndex"] = index;
        }
        if (!proposal_path_.isEmpty()) {
            row["isProposalSource"] = proposal_path_.front() == p;
            row["isProposalDest"] = proposal_path_.back() == p;
        }
        board.push_back(row);
    }
    board_model_->setRows(board);
    if (rebuildPieces) rebuildPieceModel();
    history_model_->setRows(history_);

    QVariantList players;
    for (int seat = 0; seat < match_.count; ++seat) {
        const auto& spec = match_.players[seat];
        int home_count = 0;
        for (uint8_t position : soo::topology().camp_positions[spec.target_camp])
            if (state_.occupancy[position] == spec.id) ++home_count;
        int place = 0;
        for (int index = 0; index < state_.finished_count; ++index)
            if (state_.finish_order[index] == spec.id) place = index + 1;
        const QString place_label = place == 1 ? QStringLiteral("1st")
            : place == 2 ? QStringLiteral("2nd") : place == 3 ? QStringLiteral("3rd") : QString();
        const bool is_ai = ai_seats_.contains(spec.id);
        players.push_back(QVariantMap{{"playerId", spec.id}, {"name", playerName(spec.id)},
            {"kindLabel", is_ai ? aiAgentName() : QStringLiteral("Human")},
            {"color", playerColor(spec.id)}, {"isCurrent", state_.current_player == spec.id},
            {"isAi", is_ai}, {"homeCount", home_count}, {"campSize", soo::kCampSize},
            {"hasFinished", home_count == soo::kCampSize}, {"turnIndex", seat + 1},
            {"place", place}, {"placeLabel", place_label}});
    }
    player_model_->setRows(players);
}

void NativeController::clearProposal() {
    proposal_action_ = -1;
    proposal_path_.clear();
    proposal_is_ai_ = false;
}

void NativeController::proposeAction(int32_t action, bool isAi) {
    std::vector<uint8_t> path;
    if (!soo::canonical_move_path(state_, action / soo::kBoardSize,
                                  action % soo::kBoardSize, path)) return;
    proposal_action_ = action;
    proposal_path_.clear();
    for (uint8_t position : path) proposal_path_.push_back(position);
    proposal_is_ai_ = isAi;
    status_message_ = isAi ? QStringLiteral("AI move proposed.")
                           : QStringLiteral("Move proposed. Confirm or cancel.");
    if (!isAi) ai_details_.clear();
    refreshModels();
    Q_EMIT changed();
}

void NativeController::commitAction(int32_t action) {
    const int source = action / soo::kBoardSize;
    const int destination = action % soo::kBoardSize;
    const uint8_t player = state_.current_player;
    const QVector<int> path = proposal_path_;
    const bool was_ai = proposal_is_ai_;
    const QString path_text = proposalPath();
    QVariantList path_ids;
    for (int position : path) path_ids.push_back(position);
    std::vector<uint8_t> canonical_path;
    uint8_t move_kind = soo::kStep;
    if (!soo::canonical_move_path(state_, source, destination, canonical_path, &move_kind)) {
        fail(QStringLiteral("Cannot commit an invalid move path."));
        return;
    }
    const int piece_row = piece_model_->rowWithValue(QByteArrayLiteral("positionId"), source);
    state_history_.push_back(state_);
    state_ = soo::apply_action(state_, match_, action);
    history_.push_back(QVariantMap{{"turnNumber", state_.turn_number - 1},
        {"playerId", player}, {"playerLabel", was_ai ? QStringLiteral("AI")
                                                       : QStringLiteral("P%1").arg(player)},
        {"playerColor", playerColor(player)}, {"isAi", was_ai},
        {"moveText", QStringLiteral("%1 → %2").arg(source).arg(destination)},
        {"pathText", path_text}, {"pathIds", path_ids}, {"source", source},
        {"destination", destination}, {"kind", move_kind == soo::kJump
                                                       ? QStringLiteral("jump")
                                                       : QStringLiteral("step")},
        {"hopCount", std::max(1, static_cast<int>(path.size()) - 1)}});
    last_action_ = action;
    status_message_ = QStringLiteral("Move committed.");
    selected_position_ = -1;
    legal_actions_.clear();
    clearProposal();
    refreshModels(false);
    startAnimation(piece_row, path);
}

void NativeController::startAnimation(int pieceRow, const QVector<int>& path) {
    if (pieceRow < 0 || path.size() < 2) {
        sound_player_->play();
        rebuildPieceModel();
        finishMove();
        return;
    }
    animation_row_ = pieceRow;
    animation_path_ = path;
    animation_index_ = 0;
    animating_ = true;
    status_message_ = QStringLiteral("Animating move…");
    Q_EMIT changed();
    animation_timer_->start();
}

void NativeController::animationTick() {
    ++animation_index_;
    if (animation_index_ >= animation_path_.size()) {
        stopAnimation();
        rebuildPieceModel();
        finishMove();
        return;
    }
    const int position = animation_path_[animation_index_];
    const QVariantMap point = geometry_->holes().at(position).toMap();
    const bool last_hop = animation_index_ == animation_path_.size() - 1;
    piece_model_->updateRow(animation_row_, QVariantMap{{"positionId", position},
        {"unitX", point.value("x")}, {"unitY", point.value("y")},
        {"isMoving", !last_hop}});
    sound_player_->play();
}

void NativeController::stopAnimation() {
    animation_timer_->stop();
    animation_row_ = -1;
    animation_path_.clear();
    animation_index_ = 0;
    animating_ = false;
}

void NativeController::finishMove() {
    announceFinishers();
    if (isGameOver()) {
        ai_status_ = QStringLiteral("Idle");
        status_message_ = QStringLiteral("Game over — %1").arg(resultSummary());
        Q_EMIT changed();
        Q_EMIT gameFinished(winnerId());
        return;
    }
    if (ai_seats_.contains(state_.current_player)) {
        ai_rejected_.clear();
        ai_failure_latched_ = false;
        startAiTurn();
        return;
    }
    ai_status_ = QStringLiteral("Ready");
    status_message_ = QStringLiteral("%1 to move.").arg(currentPlayerName());
    Q_EMIT changed();
}

void NativeController::announceFinishers() {
    for (int index = 0; index < state_.finished_count; ++index) {
        const int player = state_.finish_order[index];
        if (announced_finishers_.contains(player)) continue;
        announced_finishers_.push_back(player);
        Q_EMIT playerFinished(player, index + 1);
    }
}

void NativeController::fail(const QString& message) {
    error_message_ = message;
    Q_EMIT errorRaised(message);
    Q_EMIT changed();
}

void NativeController::startAiTurn() {
    if (isGameOver() || !ai_seats_.contains(state_.current_player) || ai_failure_latched_) return;
    if (ai_worker_->isRunning()) {
        ai_restart_when_idle_ = true;
        return;
    }
    ai_restart_when_idle_ = false;

    std::vector<int32_t> legal;
    soo::legal_action_ids(state_, legal);
    if (legal.empty()) {
        ai_status_ = QStringLiteral("No legal move");
        ai_thinking_ = false;
        Q_EMIT changed();
        return;
    }

    const soo::State search_state = state_;
    const soo::Match search_match = match_;
    const QVector<int32_t> rejected = ai_rejected_;
    const int simulations = ai_simulations_;
    ai_thinking_ = true;
    ai_started_at_ms_ = QDateTime::currentMSecsSinceEpoch();
    ai_status_ = QStringLiteral("Thinking…");
    status_message_ = QStringLiteral("%1 is thinking…").arg(currentPlayerName());
    ai_details_ = {QVariantMap{{"label", QStringLiteral("Agent")}, {"value", aiAgentName()}},
        QVariantMap{{"label", QStringLiteral("Legal moves")},
                    {"value", QString::number(static_cast<int>(legal.size()))}}};
#ifdef DIAMOND_QT_HAS_SOO
    if (search_match.count == 2)
        ai_details_.push_back(QVariantMap{{"label", QStringLiteral("Simulations")},
                                          {"value", QString::number(simulations)}});
#endif
    Q_EMIT changed();

    const quint64 request_generation = ++generation_;
    ++ai_search_start_count_;
    ai_worker_->start(request_generation, [search_state, search_match, rejected, legal, simulations]() -> int {
        auto is_rejected = [&rejected](int32_t action) {
            return std::find(rejected.cbegin(), rejected.cend(), action) != rejected.cend();
        };
#ifdef DIAMOND_QT_HAS_SOO
        if (search_match.count == 2) {
            configure_torch_cpu();
            QString root = QDir(QCoreApplication::applicationDirPath())
                               .filePath(QStringLiteral("artifacts/soo-spike"));
            if (!QDir(root).exists())
                root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
            const auto artifact = diamond_model::validate_soo_deployment_artifact(root.toStdString());
            diamond_model::SooModel model(artifact.width, artifact.residual_blocks);
            model->load_weights(artifact.weights);
            diamond_model::SooEvaluator evaluator(model);
            soo::MCTSConfig config;
            config.simulations = simulations;
            config.c_puct = 1.5;
            config.dirichlet_epsilon = 0.0;
            soo::MCTS2P search(search_match, evaluator, config);
            const auto result = search.run(search_state, 0.0, false);

            std::vector<size_t> ranked(result.root_actions.size());
            std::iota(ranked.begin(), ranked.end(), size_t{0});
            std::sort(ranked.begin(), ranked.end(), [&result](size_t left, size_t right) {
                if (result.visit_counts[left] != result.visit_counts[right])
                    return result.visit_counts[left] > result.visit_counts[right];
                return result.root_actions[left] < result.root_actions[right];
            });
            for (size_t index : ranked) {
                const int32_t physical_action = soo::to_physical_action(
                    result.root_actions[index], search_match, search_state.current_player);
                if (!is_rejected(physical_action)) return physical_action;
            }
            return soo::to_physical_action(
                result.selected_action, search_match, search_state.current_player);
        }
#endif
        for (int32_t action : legal) if (!is_rejected(action)) return action;
        return legal.front();
    });
}

void NativeController::selectPosition(int position) {
    if (!canSelect()) {
        fail(QStringLiteral("The board is locked right now."));
        return;
    }
    if (!soo::mutable_topology().configured || position < 0 || position >= soo::kBoardSize) return;
    if (proposal_action_ >= 0) {
        const int32_t action = selected_position_ * soo::kBoardSize + position;
        if (std::find(legal_actions_.begin(), legal_actions_.end(), action) != legal_actions_.end()) {
            proposeAction(action, false);
            return;
        }
        if (state_.occupancy[position] == state_.current_player) {
            clearProposal();
            selected_position_ = position;
        } else {
            return;
        }
    }
    if (selected_position_ < 0) {
        if (state_.occupancy[position] == state_.current_player) selected_position_ = position;
    } else {
        const int32_t action = selected_position_ * soo::kBoardSize + position;
        if (std::find(legal_actions_.begin(), legal_actions_.end(), action) != legal_actions_.end()) {
            proposeAction(action, false);
            return;
        } else if (state_.occupancy[position] == state_.current_player) selected_position_ = position;
        else return;
    }
    legal_actions_.clear();
    if (selected_position_ >= 0) {
        std::vector<int32_t> all; soo::legal_action_ids(state_, all);
        for (int32_t action : all) if (action / soo::kBoardSize == selected_position_) legal_actions_.push_back(action);
    }
    refreshModels(); Q_EMIT changed();
}

void NativeController::confirmProposal() {
    if (proposal_action_ < 0) return;
    commitAction(proposal_action_);
}

void NativeController::cancelProposal() {
    if (proposal_is_ai_) return;
    clearProposal(); selected_position_ = -1; legal_actions_.clear();
    status_message_ = QStringLiteral("Proposal cancelled.");
    refreshModels(); Q_EMIT changed();
}

void NativeController::thinkAgain() {
    if (!proposal_is_ai_ || proposal_action_ < 0 || ai_thinking_) return;
    ai_rejected_.push_back(proposal_action_);
    clearProposal();
    ai_failure_latched_ = false;
    status_message_ = QStringLiteral("Asking the AI for another move…");
    refreshModels();
    startAiTurn();
}

void NativeController::undoLastMove() {
    if (state_history_.isEmpty()) {
        fail(QStringLiteral("Nothing to undo."));
        return;
    }
    cancelSearch();
    ai_failure_latched_ = false;
    stopAnimation();
    state_ = state_history_.takeLast(); if (!history_.isEmpty()) history_.removeLast();
    ai_rejected_.clear();
    last_action_ = history_.isEmpty() ? -1
        : history_.constLast().toMap().value("source").toInt() * soo::kBoardSize
          + history_.constLast().toMap().value("destination").toInt();
    announced_finishers_.clear();
    for (int index = 0; index < state_.finished_count; ++index)
        announced_finishers_.push_back(state_.finish_order[index]);
    selected_position_ = -1; legal_actions_.clear(); clearProposal();
    status_message_ = QStringLiteral("Last move undone.");
    refreshModels(); Q_EMIT changed();
    if (ai_seats_.contains(state_.current_player)) QTimer::singleShot(0, this, &NativeController::startAiTurn);
}

void NativeController::requestAiMove() {
    if (!ai_thinking_ && proposal_action_ < 0 && isCurrentPlayerAi()) {
        ai_failure_latched_ = false;
        startAiTurn();
    }
}

void NativeController::shutdown() {
    stopAnimation();
    cancelSearch();
}

bool NativeController::gameSmoke() {
    if (!nativeRulesReady()) return false;
    std::vector<int32_t> actions;
    soo::legal_action_ids(state_, actions);
    if (actions.empty()) return false;
    bool rejected_illegal = false;
    try {
        const int source = actions.front() / soo::kBoardSize;
        (void)soo::apply_action(state_, match_, source * soo::kBoardSize + source);
    } catch (const std::invalid_argument&) {
        rejected_illegal = true;
    }
    if (!rejected_illegal) return false;
    const auto before = state_;
    const auto before_history = history_.size();
    state_history_.push_back(state_);
    state_ = soo::apply_action(state_, match_, actions.front());
    const bool moved = state_.turn_number == before.turn_number + 1 &&
                       state_.current_player != before.current_player &&
                       state_ != before;
    state_ = state_history_.takeLast();
    history_.resize(before_history);
    selected_position_ = -1;
    legal_actions_.clear();
    refreshModels();
    return moved && state_ == before;
}

bool NativeController::workerSmoke() {
    if (ai_worker_->isRunning()) return false;
    ai_worker_->start(++generation_, [] {
        QThread::msleep(50);
        return 0;
    });
    ai_worker_->cancel();
    while (ai_worker_->isRunning()) QThread::msleep(5);
    return !ai_worker_->isRunning();
}

bool NativeController::failureSmoke() {
    if (ai_worker_->isRunning()) return false;
    if (!ai_seats_.contains(state_.current_player)) ai_seats_.push_back(state_.current_player);
    ai_failure_latched_ = false;
    ai_restart_when_idle_ = false;
    ai_thinking_ = true;
    ai_status_ = QStringLiteral("Thinking…");
    const quint64 request_generation = ++generation_;
    ++ai_search_start_count_;
    ai_worker_->start(request_generation, []() -> int {
        throw std::runtime_error("intentional worker failure smoke");
    });
    return true;
}

bool NativeController::sooSmoke() {
#ifdef DIAMOND_QT_HAS_SOO
    if (!nativeRulesReady()) return false;
    try {
        configure_torch_cpu();
        const QString root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
        const auto artifact = diamond_model::validate_soo_deployment_artifact(root.toStdString());
        diamond_model::SooModel model(artifact.width, artifact.residual_blocks);
        model->load_weights(artifact.weights);
        diamond_model::SooEvaluator evaluator(model);
        soo::MCTSConfig config;
        config.simulations = 2;
        config.c_puct = 1.5;
        config.dirichlet_epsilon = 0.0;
        std::vector<int32_t> human_actions;
        soo::legal_action_ids(state_, human_actions);
        if (human_actions.empty()) return false;
        const auto ai_state = soo::apply_action(state_, match_, human_actions.front());
        if (ai_state.current_player == state_.current_player) return false;
        soo::MCTS2P search(match_, evaluator, config);
        const auto result = search.run(ai_state, 0.0, false);
        const int32_t physical_action = soo::to_physical_action(
            result.selected_action, match_, ai_state.current_player);
        const auto next_state = soo::apply_action(ai_state, match_, physical_action);
        return result.selected_action >= 0 &&
               std::find(result.root_actions.begin(), result.root_actions.end(),
                         result.selected_action) != result.root_actions.end() &&
               next_state.turn_number == ai_state.turn_number + 1;
    } catch (const std::exception& error) {
        qWarning() << "Soo smoke failed:" << error.what();
        return false;
    }
#else
    return false;
#endif
}
