#include "native_controller.hpp"

#include <QDir>
#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointF>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <map>
#include <tuple>
#include <vector>

#include "soo/board.hpp"
#include "soo/rules.hpp"

#ifdef DIAMOND_QT_HAS_SOO
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

void controller_startup_trace(const char* marker) {
    QFile file(QCoreApplication::applicationDirPath() + QStringLiteral("/diamond_qt_startup.log"));
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) return;
    file.write("controller-");
    file.write(marker);
    file.write("\n");
    file.close();
}

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
    const int axes[6][2] = {{0,3},{1,3},{2,3},{0,-3},{1,-3},{2,-3}};
    for (int camp = 0; camp < 6; ++camp) {
        std::vector<std::pair<int, QPointF>> camp_candidates;
        for (const auto& point : points) {
            const int value = camp == 0 || camp == 3 ? point.x : camp == 1 || camp == 4 ? point.y : point.z;
            const bool triangle = camp < 3 ? point.x >= -3 && point.y >= -3 && point.z >= -3
                                           : point.x <= 3 && point.y <= 3 && point.z <= 3;
            if (value != axes[camp][1] || !triangle) continue;
            const auto xy = unit_xy(point);
            camp_candidates.push_back({static_cast<int>(camp_candidates.size()), xy});
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
        result.push_back(QVariantMap{
            {"inPlay", camp != 1 && camp != 4 || player_count_ == 3},
            {"color", camp == 0 || camp == 3 ? "#34C759" : camp == 1 || camp == 4 ? "#FFCC00" : "#FF3B30"},
            {"points", camp_points}});
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

NativeController::NativeController(QObject* parent) : QObject(parent) {
    controller_startup_trace("ctor-start");
    geometry_ = new GeometryModel(this);
    controller_startup_trace("geometry");
    board_model_ = new ContractListModel("board", this);
    controller_startup_trace("board-model");
    piece_model_ = new ContractListModel("piece", this);
    controller_startup_trace("piece-model");
    history_model_ = new ContractListModel("history", this);
    controller_startup_trace("history-model");
    player_model_ = new ContractListModel("players", this);
    controller_startup_trace("player-model");
    ai_worker_ = new NativeAiWorker(this);
    controller_startup_trace("worker");
    connect(ai_worker_, &NativeAiWorker::resultReady, this,
            [this](quint64 generation, int action) {
                if (generation != generation_ || !ai_thinking_) return;
                std::vector<int32_t> legal;
                soo::legal_action_ids(state_, legal);
                if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
                    ai_thinking_ = false;
                    qWarning() << "native AI returned an illegal action" << action;
                    Q_EMIT changed();
                    return;
                }
                ai_thinking_ = false;
                selected_position_ = -1;
                legal_actions_.clear();
                const int source = action / soo::kBoardSize;
                const int destination = action % soo::kBoardSize;
                state_history_.push_back(state_);
                state_ = soo::apply_action(state_, match_, action);
                history_.push_back(QVariantMap{{"turnNumber", state_.turn_number - 1},
                    {"playerLabel", playerName(2)}, {"playerColor", playerColor(2)},
                    {"moveText", QStringLiteral("%1 → %2").arg(source).arg(destination)},
                    {"pathText", QStringLiteral("%1 → %2").arg(source).arg(destination)}, {"hopCount", 1}});
                refreshModels(); Q_EMIT changed();
            });
    connect(ai_worker_, &NativeAiWorker::failed, this,
            [this](quint64 generation, const QString& message) {
                if (generation != generation_) return;
                ai_thinking_ = false;
                qWarning() << "native AI worker:" << message;
                Q_EMIT changed();
            });
    connect(ai_worker_, &NativeAiWorker::cancelled, this,
            [this](quint64 generation) {
                if (generation == generation_) { ai_thinking_ = false; Q_EMIT changed(); }
            });
    match_.count = 2;
    match_.players[0] = soo::PlayerSpec{1, 2, 5};
    match_.players[1] = soo::PlayerSpec{2, 0, 3};
    ai_seats_ = {2};
    geometry_->setPlayerCount(match_.count);
    controller_startup_trace("match-configured");
    state_.current_player = 1;
    loadTopology();
    controller_startup_trace("topology-loaded");
    if (soo::mutable_topology().configured) {
        for (int camp = 0; camp < match_.count; ++camp) {
            const auto& spec = match_.players[camp];
            for (uint8_t position : soo::topology().camp_positions[spec.camp])
                state_.occupancy[position] = spec.id;
        }
    }
    controller_startup_trace("before-refresh");
    refreshModels();
    controller_startup_trace("after-refresh");
}

NativeController::~NativeController() { cancelSearch(); }

void NativeController::cancelSearch() {
    ++generation_;
    if (ai_worker_) ai_worker_->cancel();
    ai_thinking_ = false;
}

QUrl NativeController::defaultSaveDir() const {
    return QUrl::fromLocalFile(QDir::homePath() + QStringLiteral("/.alphadiamond/saves"));
}

QVariantList NativeController::standings() const {
    QVariantList result;
    for (int seat = 0; seat < match_.count; ++seat) {
        const auto id = match_.players[seat].id;
        int place = 0;
        for (int i = 0; i < state_.finished_count; ++i)
            if (state_.finish_order[i] == id) place = i + 1;
        const QString place_label = place == 1 ? QStringLiteral("1st")
            : place == 2 ? QStringLiteral("2nd") : place == 3 ? QStringLiteral("3rd") : QString();
        result.push_back(QVariantMap{{"place", place}, {"placeLabel", place_label},
            {"name", playerName(id)}, {"color", playerColor(id)},
            {"isAi", ai_seats_.contains(id)}});
    }
    return result;
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

void NativeController::startMatch(const QVariantList& order, const QVariantList& aiSeats) {
    if (order.size() < 2 || order.size() > 3) return;
    cancelSearch();
    match_ = {};
    match_.count = static_cast<uint8_t>(order.size());
    geometry_->setPlayerCount(match_.count);
    const int camps2[2] = {2, 0};
    const int targets2[2] = {5, 3};
    const int camps3[3] = {2, 1, 0};
    const int targets3[3] = {5, 4, 3};
    for (int i = 0; i < match_.count; ++i) {
        const int id = order.at(i).toInt();
        const int camp = match_.count == 2 ? camps2[i] : camps3[i];
        const int target = match_.count == 2 ? targets2[i] : targets3[i];
        match_.players[i] = soo::PlayerSpec{static_cast<uint8_t>(id),
                                             static_cast<uint8_t>(camp),
                                             static_cast<uint8_t>(target)};
    }
    ai_seats_ = aiSeats;
    state_ = {};
    state_.current_player = static_cast<uint8_t>(order.at(0).toInt());
    state_history_.clear(); history_.clear(); selected_position_ = -1; legal_actions_.clear();
    for (int i = 0; i < match_.count; ++i)
        for (const auto position : soo::topology().camp_positions[match_.players[i].camp])
            state_.occupancy[position] = match_.players[i].id;
    refreshModels(); Q_EMIT changed();
}

void NativeController::saveGame(const QUrl& path) {
    if (!path.isLocalFile()) return;
    QJsonObject root;
    root["version"] = 1;
    root["playerCount"] = match_.count;
    root["currentPlayer"] = state_.current_player;
    root["turnNumber"] = state_.turn_number;
    QJsonArray order, ai, occupancy;
    for (const auto& p : match_.players) if (p.id != 0) order.push_back(p.id);
    for (const auto& seat : ai_seats_) ai.push_back(seat.toInt());
    for (const auto piece : state_.occupancy) occupancy.push_back(piece);
    root["order"] = order; root["aiSeats"] = ai; root["occupancy"] = occupancy;
    QFile file(path.toLocalFile());
    if (file.open(QIODevice::WriteOnly)) file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
}

void NativeController::loadGame(const QUrl& path) {
    if (!path.isLocalFile()) return;
    QFile file(path.toLocalFile());
    if (!file.open(QIODevice::ReadOnly)) return;
    const auto root = QJsonDocument::fromJson(file.readAll()).object();
    const auto orderJson = root.value("order").toArray();
    const auto aiJson = root.value("aiSeats").toArray();
    QVariantList order, ai;
    for (const auto value : orderJson) order.push_back(value.toInt());
    for (const auto value : aiJson) ai.push_back(value.toInt());
    if (order.size() < 2 || order.size() > 3) return;
    startMatch(order, ai);
    const auto occupancy = root.value("occupancy").toArray();
    if (occupancy.size() == soo::kBoardSize) {
        for (int i = 0; i < soo::kBoardSize; ++i) state_.occupancy[i] = static_cast<uint8_t>(occupancy.at(i).toInt());
        state_.current_player = static_cast<uint8_t>(root.value("currentPlayer").toInt(state_.current_player));
        state_.turn_number = root.value("turnNumber").toInt(0);
        refreshModels(); Q_EMIT changed();
    }
}

void NativeController::loadTopology() {
    controller_startup_trace("load-topology-start");
    const QString root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
    controller_startup_trace("load-topology-root");
    auto& topo = soo::mutable_topology();
    controller_startup_trace("load-topology-state");
    {
        std::ifstream file(root.toStdString() + "/topology_neighbour.i8", std::ios::binary);
        controller_startup_trace("load-topology-neighbour-open");
        if (!file) return;
        file.read(reinterpret_cast<char*>(topo.neighbour.data()), sizeof(topo.neighbour));
        controller_startup_trace("load-topology-neighbour-read");
        if (file.gcount() != static_cast<std::streamsize>(sizeof(topo.neighbour))) return;
    }
    std::vector<int32_t> camps(60), pairwise(5329), physical(438), canonical(438);
    auto read_vector = [&](const char* name, std::vector<int32_t>& values) {
        controller_startup_trace(name);
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

void NativeController::refreshModels() {
    QVariantList board, pieces;
    const auto geo = geometry_->holes();
    for (int p = 0; p < soo::kBoardSize; ++p) {
        QVariantMap row{{"positionId", p}, {"unitX", geo[p].toMap().value("x")},
                        {"unitY", geo[p].toMap().value("y")}, {"campKey", "neutral"},
                        {"occupant", state_.occupancy[p]}, {"isSelected", p == selected_position_},
                        {"isLegalStep", false}, {"isLegalJump", false}, {"isPathNode", false},
                        {"pathIndex", -1}, {"isLastMoveSource", false}, {"isLastMoveDest", false},
                        {"isProposalSource", false}, {"isProposalDest", false}};
        for (int32_t action : legal_actions_) if (action % soo::kBoardSize == p) row["isLegalStep"] = true;
        board.push_back(row);
        if (state_.occupancy[p] != soo::kEmpty)
            pieces.push_back(QVariantMap{{"pieceId", p}, {"playerId", state_.occupancy[p]},
                {"positionId", p}, {"unitX", geo[p].toMap().value("x")},
                {"unitY", geo[p].toMap().value("y")}, {"color", playerColor(state_.occupancy[p])},
                {"isMoving", false}});
    }
    board_model_->setRows(board); piece_model_->setRows(pieces); history_model_->setRows(history_);
}

void NativeController::selectPosition(int position) {
    if (ai_thinking_) return;
    if (!soo::mutable_topology().configured || position < 0 || position >= soo::kBoardSize) return;
    if (selected_position_ < 0) {
        if (state_.occupancy[position] == state_.current_player) selected_position_ = position;
    } else {
        const int32_t action = selected_position_ * soo::kBoardSize + position;
        if (std::find(legal_actions_.begin(), legal_actions_.end(), action) != legal_actions_.end()) {
            state_history_.push_back(state_);
            state_ = soo::apply_action(state_, match_, action);
            history_.push_back(QVariantMap{{"turnNumber", state_.turn_number - 1},
                {"playerLabel", playerName(state_history_.back().current_player)},
                {"playerColor", playerColor(state_history_.back().current_player)},
                {"moveText", QStringLiteral("%1 → %2").arg(selected_position_).arg(position)},
                {"pathText", QStringLiteral("%1 → %2").arg(selected_position_).arg(position)}, {"hopCount", 1}});
            selected_position_ = -1;
            if (match_.count == 2 && ai_seats_.contains(state_.current_player) && !isGameOver()) {
                ai_thinking_ = true;
                const quint64 generation = ++generation_;
                const soo::State search_state = state_;
                const soo::Match search_match = match_;
                const QString model_root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
                ai_worker_->start(generation, [search_state, search_match, model_root] {
#ifdef DIAMOND_QT_HAS_SOO
                    configure_torch_cpu();
                    diamond_model::SooModel model(128, 6);
                    model->load_weights(model_root.toStdString() + "/weights");
                    diamond_model::SooEvaluator evaluator(model);
                    soo::MCTSConfig config;
                    config.simulations = 32;
                    config.c_puct = 1.5;
                    config.dirichlet_epsilon = 0.0;
                    soo::MCTS2P search(search_match, evaluator, config);
                    return search.run(search_state, 0.0, false).selected_action;
#else
                    std::vector<int32_t> actions;
                    soo::legal_action_ids(search_state, actions);
                    if (actions.empty()) throw std::runtime_error("native AI has no legal action");
                    return actions.front();
#endif
                });
            }
        } else if (state_.occupancy[position] == state_.current_player) selected_position_ = position;
        else selected_position_ = -1;
    }
    legal_actions_.clear();
    if (selected_position_ >= 0) {
        std::vector<int32_t> all; soo::legal_action_ids(state_, all);
        for (int32_t action : all) if (action / soo::kBoardSize == selected_position_) legal_actions_.push_back(action);
    }
    refreshModels(); Q_EMIT changed();
}

void NativeController::cancelProposal() { selected_position_ = -1; legal_actions_.clear(); refreshModels(); Q_EMIT changed(); }

void NativeController::undoLastMove() {
    if (state_history_.isEmpty()) return;
    state_ = state_history_.takeLast(); if (!history_.isEmpty()) history_.removeLast();
    selected_position_ = -1; legal_actions_.clear(); refreshModels(); Q_EMIT changed();
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

bool NativeController::sooSmoke() {
#ifdef DIAMOND_QT_HAS_SOO
    if (!nativeRulesReady()) return false;
    try {
        configure_torch_cpu();
        diamond_model::SooModel model(128, 6);
        const QString root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
        model->load_weights(root.toStdString() + "/weights");
        diamond_model::SooEvaluator evaluator(model);
        soo::MCTSConfig config;
        config.simulations = 2;
        config.c_puct = 1.5;
        config.dirichlet_epsilon = 0.0;
        soo::MCTS2P search(match_, evaluator, config);
        const auto result = search.run(state_, 0.0, false);
        return result.selected_action >= 0 &&
               std::find(result.root_actions.begin(), result.root_actions.end(),
                         result.selected_action) != result.root_actions.end();
    } catch (const std::exception& error) {
        qWarning() << "Soo smoke failed:" << error.what();
        return false;
    }
#else
    return false;
#endif
}
