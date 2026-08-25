#pragma once

#include <QAbstractListModel>
#include <QColor>
#include <QObject>
#include <QUrl>
#include <QVariantList>
#include <QVector>

#include <algorithm>
#include <memory>
#include <optional>

#include "soo/state.hpp"
#include "ai_worker.hpp"

class QTimer;
class NativeMovePlayer;
class SooSearchRuntime;

class GeometryModel final : public QObject {
    Q_OBJECT
  public:
    explicit GeometryModel(QObject* parent = nullptr);
    void setPlayerCount(int count);

    Q_INVOKABLE QVariantMap bounds() const;
    Q_INVOKABLE QVariantList edges() const;
    Q_INVOKABLE QVariantList camps() const;
    Q_INVOKABLE QVariantList holes() const;

  Q_SIGNALS:
    void changed();

  private:
    int player_count_ = 3;
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
        PathTextRole, HopCountRole, IsAiRole, NameRole, KindLabelRole,
        IsCurrentRole, HomeCountRole, CampSizeRole, HasFinishedRole,
        TurnIndexRole, PlaceRole, PlaceLabelRole
    };

    explicit ContractListModel(QString kind, QObject* parent = nullptr);
    int rowCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;
    void setRows(QVariantList rows);
    const QVariantList& rows() const { return rows_; }
    int rowWithValue(const QByteArray& roleName, const QVariant& value) const;
    void updateRow(int row, const QVariantMap& values);

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
    Q_PROPERTY(QString phase READ phase NOTIFY changed)
    Q_PROPERTY(QString gameLabel READ gameLabel NOTIFY changed)
    Q_PROPERTY(int playerCount READ playerCount NOTIFY changed)
    Q_PROPERTY(int turnNumber READ turnNumber NOTIFY changed)
    Q_PROPERTY(QString currentPlayerName READ currentPlayerName NOTIFY changed)
    Q_PROPERTY(QString currentPlayerColor READ currentPlayerColor NOTIFY changed)
    Q_PROPERTY(int currentPlayerId READ currentPlayerId NOTIFY changed)
    Q_PROPERTY(bool isCurrentPlayerAi READ isCurrentPlayerAi NOTIFY changed)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY changed)
    Q_PROPERTY(QString errorMessage READ errorMessage NOTIFY changed)
    Q_PROPERTY(bool isGameOver READ isGameOver NOTIFY changed)
    Q_PROPERTY(int winnerId READ winnerId NOTIFY changed)
    Q_PROPERTY(QString winnerName READ winnerName NOTIFY changed)
    Q_PROPERTY(QString resultSummary READ resultSummary NOTIFY changed)
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
    Q_PROPERTY(int proposalHopCount READ proposalHopCount NOTIFY changed)
    Q_PROPERTY(QString aiAgentName READ aiAgentName NOTIFY changed)
    Q_PROPERTY(QString aiStatus READ aiStatus NOTIFY changed)
    Q_PROPERTY(QVariantList aiDetails READ aiDetails NOTIFY changed)
    Q_PROPERTY(QString lastMoveText READ lastMoveText NOTIFY changed)
    Q_PROPERTY(int selectedPosition READ selectedPosition NOTIFY changed)
    Q_PROPERTY(bool soundAvailable READ soundAvailable NOTIFY changed)
    Q_PROPERTY(bool soundEnabled READ soundEnabled NOTIFY changed)
    Q_PROPERTY(QString soundStatus READ soundStatus NOTIFY changed)
    Q_PROPERTY(double soundVolume READ soundVolume NOTIFY changed)
    Q_PROPERTY(QUrl defaultSaveDir READ defaultSaveDir CONSTANT)
    Q_PROPERTY(QVariantList standings READ standings NOTIFY changed)
    Q_PROPERTY(QVariantList turnOrder READ turnOrder NOTIFY changed)
    Q_PROPERTY(QVariantList aiSeats READ aiSeats NOTIFY changed)
    Q_PROPERTY(bool nativeRulesReady READ nativeRulesReady CONSTANT)
    Q_PROPERTY(bool aiThinking READ aiThinking NOTIFY changed)
    Q_PROPERTY(QVariantList positionTelemetry READ positionTelemetry NOTIFY changed)
    Q_PROPERTY(QVariantList decisionTelemetry READ decisionTelemetry NOTIFY changed)
    Q_PROPERTY(QVariantMap latestSearchCompute READ latestSearchCompute NOTIFY changed)
    Q_PROPERTY(bool analysisAvailable READ analysisAvailable NOTIFY changed)
    Q_PROPERTY(int perspectivePlayerId READ perspectivePlayerId WRITE setPerspectivePlayerId NOTIFY changed)

  public:
    explicit NativeController(QObject* parent = nullptr);
    ~NativeController() override;

    QObject* boardModel() const { return board_model_; }
    QObject* pieceModel() const { return piece_model_; }
    QObject* historyModel() const { return history_model_; }
    QObject* playerModel() const { return player_model_; }
    QObject* geometry() const { return geometry_; }
    QString phase() const;
    QString gameLabel() const { return QStringLiteral("Game #%1").arg(game_number_, 3, 10, QLatin1Char('0')); }
    int playerCount() const { return match_.count; }
    int turnNumber() const { return state_.turn_number; }
    QString currentPlayerName() const;
    QString currentPlayerColor() const;
    int currentPlayerId() const { return state_.current_player; }
    bool isCurrentPlayerAi() const { return ai_seats_.contains(state_.current_player); }
    QString statusMessage() const { return status_message_; }
    QString errorMessage() const { return error_message_; }
    bool isGameOver() const { return state_.status == soo::kFinished; }
    int winnerId() const { return state_.finished_count ? state_.finish_order[0] : 0; }
    QString winnerName() const { return winnerId() ? playerName(static_cast<uint8_t>(winnerId())) : QString(); }
    QString resultSummary() const;
    bool canSelect() const { return !isGameOver() && !ai_thinking_ && !animating_ && !proposal_is_ai_; }
    bool canUndo() const { return !history_.isEmpty() && !animating_; }
    bool canConfirm() const { return proposal_action_ >= 0; }
    bool canCancel() const { return (!proposal_is_ai_ && proposal_action_ >= 0) || selected_position_ >= 0; }
    bool hasProposal() const { return proposal_action_ >= 0; }
    bool proposalIsAi() const { return proposal_is_ai_; }
    bool proposalIsMultiHop() const { return proposal_path_.size() > 2; }
    QString proposalSummary() const;
    QString proposalPath() const;
    QVariantList proposalPathIds() const;
    int proposalHopCount() const { return std::max(0, static_cast<int>(proposal_path_.size()) - 1); }
    QString aiAgentName() const;
    QString aiStatus() const { return ai_status_; }
    QVariantList aiDetails() const { return ai_details_; }
    QString lastMoveText() const;
    int selectedPosition() const { return selected_position_; }
    bool soundAvailable() const;
    bool soundEnabled() const;
    QString soundStatus() const;
    double soundVolume() const;
    bool soundLoaded() const;
    int soundPlayRequestCount() const;
    QUrl defaultSaveDir() const;
    QVariantList standings() const;
    QVariantList turnOrder() const;
    QVariantList aiSeats() const;
    bool nativeRulesReady() const { return soo::mutable_topology().configured; }
    bool aiThinking() const { return ai_thinking_; }
    int aiSearchStartCount() const { return ai_search_start_count_; }
    QVariantList positionTelemetry() const;
    QVariantList decisionTelemetry() const;
    QVariantMap latestSearchCompute() const { return latest_search_compute_; }
    bool analysisAvailable() const;
    int perspectivePlayerId() const { return perspective_player_id_; }

    Q_INVOKABLE QVariantList seatColorsFor(int count) const;
    Q_INVOKABLE void selectPosition(int position);
    Q_INVOKABLE void confirmProposal();
    Q_INVOKABLE void cancelProposal();
    Q_INVOKABLE void undoLastMove();
    Q_INVOKABLE void thinkAgain();
    Q_INVOKABLE void previewSound();
    Q_INVOKABLE void setSoundEnabled(bool enabled);
    Q_INVOKABLE void setSoundVolume(double volume);
    Q_INVOKABLE void newGame();
    Q_INVOKABLE bool startMatch(const QVariantList& order, const QVariantList& aiSeats);
    Q_INVOKABLE bool saveGame(const QUrl& path);
    Q_INVOKABLE bool loadGame(const QUrl& path);
    Q_INVOKABLE void requestAiMove();
    Q_INVOKABLE void shutdown();
    Q_INVOKABLE bool gameSmoke();
    Q_INVOKABLE bool workerSmoke();
    Q_INVOKABLE bool failureSmoke();
    Q_INVOKABLE bool sooSmoke();
    Q_INVOKABLE void setPerspectivePlayerId(int playerId);

  Q_SIGNALS:
    void changed();
    void errorRaised(const QString& message);
    void gameFinished(int winnerId);
    void playerFinished(int playerId, int place);

  private:
    void loadTopology();
    void refreshModels(bool rebuildPieces = true);
    void rebuildPieceModel();
    void cancelSearch();
    void proposeAction(int32_t action, bool isAi);
    void clearProposal();
    void commitAction(int32_t action);
    void startAnimation(int pieceRow, const QVector<int>& path);
    void animationTick();
    void stopAnimation();
    void finishMove();
    void startAiTurn();
    void startHumanAnalysis();
    void startSearch(bool selectMove);
    void appendTelemetryForCommit(uint8_t player, int32_t action);
    void publishLatestCompute(const SearchTelemetry& telemetry);
    void announceFinishers();
    void fail(const QString& message);
    QString playerColor(uint8_t id) const;
    QString playerName(uint8_t id) const;

    soo::Match match_;
    soo::State state_;
    QVector<soo::State> state_history_;
    QVariantList history_;
    QVector<int32_t> legal_actions_;
    int32_t proposal_action_ = -1;
    QVector<int> proposal_path_;
    bool proposal_is_ai_ = false;
    int next_piece_id_ = 0;
    QTimer* animation_timer_ = nullptr;
    int animation_row_ = -1;
    QVector<int> animation_path_;
    int animation_index_ = 0;
    bool animating_ = false;
    int game_number_ = 1;
    int32_t last_action_ = -1;
    QVector<int> announced_finishers_;
    QString status_message_ = QStringLiteral("Ready.");
    QString error_message_;
    QVariantList ai_seats_;
    QVector<int32_t> ai_rejected_;
    QString ai_status_ = QStringLiteral("Ready");
    QVariantList ai_details_;
    int ai_simulations_ = 1024;
    qint64 ai_started_at_ms_ = 0;
    int selected_position_ = -1;
    bool ai_thinking_ = false;
    bool ai_restart_when_idle_ = false;
    bool ai_failure_latched_ = false;
    int ai_search_start_count_ = 0;
    quint64 generation_ = 0;
    NativeAiWorker* ai_worker_;
    // A running worker keeps the runtime alive even if the window/controller
    // is closed while a search is still in flight.
    std::shared_ptr<SooSearchRuntime> soo_runtime_;
    NativeMovePlayer* sound_player_;
    GeometryModel* geometry_;
    ContractListModel* board_model_;
    ContractListModel* piece_model_;
    ContractListModel* history_model_;
    ContractListModel* player_model_;
    enum class SearchPurpose : uint8_t { None, AiMove, HumanAnalysis };
    SearchPurpose search_purpose_ = SearchPurpose::None;
    bool analysis_thinking_ = false;
    std::optional<SearchTelemetry> pending_telemetry_;
    int pending_telemetry_turn_ = -1;
    int pending_telemetry_player_ = 0;
    QVariantList position_telemetry_;
    QVariantList decision_telemetry_;
    QVariantMap latest_search_compute_;
    int perspective_player_id_ = 1;
};
