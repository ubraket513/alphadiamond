#pragma once

#include <QAbstractListModel>
#include <QColor>
#include <QObject>
#include <QUrl>
#include <QVariantList>
#include <QVector>

#include "soo/state.hpp"
#include "ai_worker.hpp"

class GeometryModel final : public QObject {
    Q_OBJECT
  public:
    explicit GeometryModel(QObject* parent = nullptr);

    Q_INVOKABLE QVariantMap bounds() const;
    Q_INVOKABLE QVariantList edges() const;
    Q_INVOKABLE QVariantList camps() const;
    Q_INVOKABLE QVariantList holes() const;
};

class ContractListModel final : public QAbstractListModel {
    Q_OBJECT
  public:
    enum Role {
        PositionIdRole = Qt::UserRole + 1,
        UnitXRole, UnitYRole, CampKeyRole, OccupantRole,
        IsSelectedRole, IsLegalStepRole, IsLegalJumpRole, IsPathNodeRole,
        PathIndexRole, IsLastMoveSourceRole, IsLastMoveDestRole,
        IsProposalSourceRole, IsProposalDestRole,
        PieceIdRole, PlayerIdRole, ColorRole, IsMovingRole,
        TurnNumberRole, PlayerLabelRole, PlayerColorRole, MoveTextRole,
        PathTextRole, HopCountRole
    };

    explicit ContractListModel(QString kind, QObject* parent = nullptr);
    int rowCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;
    void setRows(QVariantList rows);

  private:
    QString kind_;
    QVariantList rows_;
    QHash<int, QByteArray> roles_;
};

class NativeController final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QObject* boardModel READ boardModel CONSTANT)
    Q_PROPERTY(QObject* pieceModel READ pieceModel CONSTANT)
    Q_PROPERTY(QObject* historyModel READ historyModel CONSTANT)
    Q_PROPERTY(QObject* playerModel READ playerModel CONSTANT)
    Q_PROPERTY(QObject* geometry READ geometry CONSTANT)
    Q_PROPERTY(QString gameLabel READ gameLabel NOTIFY changed)
    Q_PROPERTY(int playerCount READ playerCount NOTIFY changed)
    Q_PROPERTY(int turnNumber READ turnNumber NOTIFY changed)
    Q_PROPERTY(QString currentPlayerName READ currentPlayerName NOTIFY changed)
    Q_PROPERTY(QString currentPlayerColor READ currentPlayerColor NOTIFY changed)
    Q_PROPERTY(bool isGameOver READ isGameOver NOTIFY changed)
    Q_PROPERTY(bool canSelect READ canSelect NOTIFY changed)
    Q_PROPERTY(bool canUndo READ canUndo NOTIFY changed)
    Q_PROPERTY(bool canConfirm READ canConfirm NOTIFY changed)
    Q_PROPERTY(bool canCancel READ canCancel NOTIFY changed)
    Q_PROPERTY(bool hasProposal READ hasProposal NOTIFY changed)
    Q_PROPERTY(bool proposalIsAi READ proposalIsAi NOTIFY changed)
    Q_PROPERTY(bool proposalIsMultiHop READ proposalIsMultiHop NOTIFY changed)
    Q_PROPERTY(QString proposalSummary READ proposalSummary NOTIFY changed)
    Q_PROPERTY(QString proposalPath READ proposalPath NOTIFY changed)
    Q_PROPERTY(QVariantList proposalPathIds READ proposalPathIds NOTIFY changed)
    Q_PROPERTY(QString aiAgentName READ aiAgentName NOTIFY changed)
    Q_PROPERTY(QString aiStatus READ aiStatus NOTIFY changed)
    Q_PROPERTY(QVariantList aiDetails READ aiDetails NOTIFY changed)
    Q_PROPERTY(int selectedPosition READ selectedPosition NOTIFY changed)
    Q_PROPERTY(bool soundAvailable READ soundAvailable CONSTANT)
    Q_PROPERTY(bool soundEnabled READ soundEnabled NOTIFY changed)
    Q_PROPERTY(QString soundStatus READ soundStatus NOTIFY changed)
    Q_PROPERTY(double soundVolume READ soundVolume NOTIFY changed)
    Q_PROPERTY(QUrl defaultSaveDir READ defaultSaveDir CONSTANT)
    Q_PROPERTY(QVariantList standings READ standings NOTIFY changed)
    Q_PROPERTY(QVariantList turnOrder READ turnOrder NOTIFY changed)
    Q_PROPERTY(QVariantList aiSeats READ aiSeats NOTIFY changed)
    Q_PROPERTY(bool nativeRulesReady READ nativeRulesReady CONSTANT)
    Q_PROPERTY(bool aiThinking READ aiThinking NOTIFY changed)

  public:
    explicit NativeController(QObject* parent = nullptr);

    QObject* boardModel() const { return board_model_; }
    QObject* pieceModel() const { return piece_model_; }
    QObject* historyModel() const { return history_model_; }
    QObject* playerModel() const { return player_model_; }
    QObject* geometry() const { return geometry_; }
    QString gameLabel() const { return QStringLiteral("Game #001"); }
    int playerCount() const { return match_.count; }
    int turnNumber() const { return state_.turn_number; }
    QString currentPlayerName() const;
    QString currentPlayerColor() const;
    bool isGameOver() const { return state_.status == soo::kFinished; }
    bool canSelect() const { return !isGameOver() && !ai_thinking_; }
    bool canUndo() const { return !history_.isEmpty(); }
    bool canConfirm() const { return false; }
    bool canCancel() const { return false; }
    bool hasProposal() const { return false; }
    bool proposalIsAi() const { return false; }
    bool proposalIsMultiHop() const { return false; }
    QString proposalSummary() const { return {}; }
    QString proposalPath() const { return {}; }
    QVariantList proposalPathIds() const { return {}; }
    QString aiAgentName() const { return QStringLiteral("Native shell placeholder"); }
    QString aiStatus() const { return ai_thinking_ ? QStringLiteral("Thinking…") : QStringLiteral("Ready"); }
    QVariantList aiDetails() const { return {}; }
    int selectedPosition() const { return selected_position_; }
    bool soundAvailable() const { return false; }
    bool soundEnabled() const { return false; }
    QString soundStatus() const { return QStringLiteral("Sound is not enabled in Q3 shell."); }
    double soundVolume() const { return 0.0; }
    QUrl defaultSaveDir() const;
    QVariantList standings() const;
    QVariantList turnOrder() const;
    QVariantList aiSeats() const { return {2}; }
    bool nativeRulesReady() const { return soo::mutable_topology().configured; }
    bool aiThinking() const { return ai_thinking_; }

    Q_INVOKABLE QVariantList seatColorsFor(int count) const;
    Q_INVOKABLE void selectPosition(int position);
    Q_INVOKABLE void confirmProposal() {}
    Q_INVOKABLE void cancelProposal();
    Q_INVOKABLE void undoLastMove();
    Q_INVOKABLE void thinkAgain() {}
    Q_INVOKABLE void previewSound() {}
    Q_INVOKABLE void setSoundEnabled(bool) {}
    Q_INVOKABLE void setSoundVolume(double) {}
    Q_INVOKABLE void startMatch(const QVariantList&, const QVariantList&) {}
    Q_INVOKABLE void saveGame(const QUrl&) {}
    Q_INVOKABLE void loadGame(const QUrl&) {}
    Q_INVOKABLE bool gameSmoke();
    Q_INVOKABLE bool workerSmoke();

  signals:
    void changed();

  private:
    void loadTopology();
    void refreshModels();
    QString playerColor(uint8_t id) const;
    QString playerName(uint8_t id) const;

    soo::Match match_;
    soo::State state_;
    QVector<soo::State> state_history_;
    QVariantList history_;
    QVector<int32_t> legal_actions_;
    int selected_position_ = -1;
    bool ai_thinking_ = false;
    quint64 generation_ = 0;
    NativeAiWorker* ai_worker_;
    GeometryModel* geometry_;
    ContractListModel* board_model_;
    ContractListModel* piece_model_;
    ContractListModel* history_model_;
    ContractListModel* player_model_;
};
