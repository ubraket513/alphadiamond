#include "native_controller.hpp"

#include <QDir>
#include <QCoreApplication>
#include <QDebug>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "soo/board.hpp"
#include "soo/rules.hpp"

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/soo_evaluator.hpp"
#include "soo/mcts.hpp"
#endif

GeometryModel::GeometryModel(QObject* parent) : QObject(parent) {}

QVariantMap GeometryModel::bounds() const {
    return {{"minX", -6.0}, {"minY", -6.0}, {"maxX", 6.0}, {"maxY", 6.0}};
}

QVariantList GeometryModel::holes() const {
    QVariantList result;
    for (int i = 0; i < 73; ++i) {
        const double angle = (2.0 * M_PI * i) / 73.0;
        result.push_back(QVariantMap{{"x", 5.2 * std::cos(angle)},
                                     {"y", 5.2 * std::sin(angle)}});
    }
    return result;
}

QVariantList GeometryModel::edges() const {
    QVariantList result;
    for (int i = 0; i < 72; ++i) {
        const auto a = holes().at(i).toMap();
        const auto b = holes().at(i + 1).toMap();
        result.push_back(QVariantMap{{"x1", a.value("x")}, {"y1", a.value("y")},
                                     {"x2", b.value("x")}, {"y2", b.value("y")}});
    }
    return result;
}

QVariantList GeometryModel::camps() const {
    QVariantList result;
    for (int camp = 0; camp < 6; ++camp) {
        const double angle = camp * M_PI / 3.0;
        result.push_back(QVariantMap{
            {"inPlay", camp < 3},
            {"color", camp == 0 ? "#FF3B30" : camp == 1 ? "#FFCC00" : "#34C759"},
            {"points", QVariantList{
                QVariantMap{{"x", 5.6 * std::cos(angle)}, {"y", 5.6 * std::sin(angle)}},
                QVariantMap{{"x", 3.4 * std::cos(angle + 0.35)}, {"y", 3.4 * std::sin(angle + 0.35)}},
                QVariantMap{{"x", 3.4 * std::cos(angle - 0.35)}, {"y", 3.4 * std::sin(angle - 0.35)}}
            }}});
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
    geometry_ = new GeometryModel(this);
    board_model_ = new ContractListModel("board", this);
    piece_model_ = new ContractListModel("piece", this);
    history_model_ = new ContractListModel("history", this);
    player_model_ = new ContractListModel("players", this);
    ai_worker_ = new NativeAiWorker(this);
    connect(ai_worker_, &NativeAiWorker::resultReady, this,
            [this](quint64 generation, int action) {
                if (generation != generation_ || !ai_thinking_) return;
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
                refreshModels(); emit changed();
            });
    connect(ai_worker_, &NativeAiWorker::failed, this,
            [this](quint64 generation, const QString& message) {
                if (generation != generation_) return;
                ai_thinking_ = false;
                qWarning() << "native AI worker:" << message;
                emit changed();
            });
    connect(ai_worker_, &NativeAiWorker::cancelled, this,
            [this](quint64 generation) {
                if (generation == generation_) { ai_thinking_ = false; emit changed(); }
            });
    match_.count = 2;
    match_.players[0] = soo::PlayerSpec{1, 2, 5};
    match_.players[1] = soo::PlayerSpec{2, 0, 3};
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

QUrl NativeController::defaultSaveDir() const {
    return QUrl::fromLocalFile(QDir::homePath() + QStringLiteral("/.alphadiamond/saves"));
}

QVariantList NativeController::standings() const {
    return {QVariantMap{{"place", 1}, {"placeLabel", "1st"}, {"name", "Player 1"},
                        {"color", "#FF3B30"}, {"isAi", false}}};
}

QVariantList NativeController::seatColorsFor(int count) const {
    if (count == 2) return {"#FF3B30", "#34C759"};
    return {"#FF3B30", "#FFCC00", "#34C759"};
}

QString NativeController::playerColor(uint8_t id) const {
    return id == 1 ? QStringLiteral("#FF3B30") : QStringLiteral("#34C759");
}

QString NativeController::playerName(uint8_t id) const { return QStringLiteral("Player %1").arg(id); }

QString NativeController::currentPlayerName() const { return playerName(state_.current_player); }
QString NativeController::currentPlayerColor() const { return playerColor(state_.current_player); }

QVariantList NativeController::turnOrder() const { return {1, 2}; }

void NativeController::loadTopology() {
    const QString root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
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
            if (state_.current_player == 2 && !isGameOver()) {
                ai_thinking_ = true;
                const quint64 generation = ++generation_;
                const soo::State search_state = state_;
                const soo::Match search_match = match_;
                const QString model_root = QDir::current().filePath(QStringLiteral("artifacts/soo-spike"));
                ai_worker_->start(generation, [search_state, search_match, model_root] {
#ifdef DIAMOND_QT_HAS_SOO
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
    refreshModels(); emit changed();
}

void NativeController::cancelProposal() { selected_position_ = -1; legal_actions_.clear(); refreshModels(); emit changed(); }

void NativeController::undoLastMove() {
    if (state_history_.isEmpty()) return;
    state_ = state_history_.takeLast(); if (!history_.isEmpty()) history_.removeLast();
    selected_position_ = -1; legal_actions_.clear(); refreshModels(); emit changed();
}

bool NativeController::gameSmoke() {
    if (!nativeRulesReady()) return false;
    std::vector<int32_t> actions;
    soo::legal_action_ids(state_, actions);
    if (actions.empty()) return false;
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
