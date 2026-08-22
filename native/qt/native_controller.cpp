#include "native_controller.hpp"

#include <QDir>

#include <cmath>

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
    : QAbstractListModel(parent), kind_(std::move(kind)) {}

int ContractListModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    if (kind_ == "board") return 73;
    if (kind_ == "piece") return 30;
    return 0;
}

QVariant ContractListModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid()) return {};
    const int row = index.row();
    if (kind_ == "board") {
        const double angle = 2.0 * M_PI * row / 73.0;
        switch (role) {
            case PositionIdRole: return row;
            case UnitXRole: return 5.2 * std::cos(angle);
            case UnitYRole: return 5.2 * std::sin(angle);
            case CampKeyRole: return QStringLiteral("neutral");
            case OccupantRole: return 0;
            case IsSelectedRole: case IsLegalStepRole: case IsLegalJumpRole:
            case IsPathNodeRole: case IsLastMoveSourceRole: case IsLastMoveDestRole:
            case IsProposalSourceRole: case IsProposalDestRole: return false;
            case PathIndexRole: return -1;
            default: return {};
        }
    }
    if (kind_ == "piece") {
        const int player = row / 10 + 1;
        switch (role) {
            case PieceIdRole: return row;
            case PlayerIdRole: return player;
            case PositionIdRole: return row;
            case UnitXRole: return 5.2 * std::cos(2.0 * M_PI * row / 30.0);
            case UnitYRole: return 5.2 * std::sin(2.0 * M_PI * row / 30.0);
            case ColorRole: return player == 1 ? "#FF3B30" : player == 2 ? "#FFCC00" : "#34C759";
            case IsMovingRole: return false;
            default: return {};
        }
    }
    return {};
}

QHash<int, QByteArray> ContractListModel::roleNames() const {
    if (kind_ == "board") return {
        {PositionIdRole, "positionId"}, {UnitXRole, "unitX"}, {UnitYRole, "unitY"},
        {CampKeyRole, "campKey"}, {OccupantRole, "occupant"}, {IsSelectedRole, "isSelected"},
        {IsLegalStepRole, "isLegalStep"}, {IsLegalJumpRole, "isLegalJump"},
        {IsPathNodeRole, "isPathNode"}, {PathIndexRole, "pathIndex"},
        {IsLastMoveSourceRole, "isLastMoveSource"}, {IsLastMoveDestRole, "isLastMoveDest"},
        {IsProposalSourceRole, "isProposalSource"}, {IsProposalDestRole, "isProposalDest"}};
    if (kind_ == "piece") return {
        {PieceIdRole, "pieceId"}, {PlayerIdRole, "playerId"}, {PositionIdRole, "positionId"},
        {UnitXRole, "unitX"}, {UnitYRole, "unitY"}, {ColorRole, "color"}, {IsMovingRole, "isMoving"}};
    return {};
}

NativeController::NativeController(QObject* parent) : QObject(parent) {
    geometry_ = new GeometryModel(this);
    board_model_ = new ContractListModel("board", this);
    piece_model_ = new ContractListModel("piece", this);
    history_model_ = new ContractListModel("history", this);
    player_model_ = new ContractListModel("players", this);
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

void NativeController::selectPosition(int) { emit changed(); }
