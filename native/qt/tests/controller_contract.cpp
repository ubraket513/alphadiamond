#include <QAbstractItemModel>
#include <QElapsedTimer>
#include <QGuiApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <QTemporaryDir>

#include <cstdlib>
#include <cmath>
#include <functional>

#include "native_controller.hpp"

namespace {

int role_for(const QAbstractItemModel* model, const QByteArray& name) {
    const auto roles = model->roleNames();
    for (auto it = roles.cbegin(); it != roles.cend(); ++it) {
        if (it.value() == name) return it.key();
    }
    return -1;
}

bool require(bool condition, const char* message) {
    if (!condition) qCritical("controller contract failed: %s", message);
    return condition;
}

bool has_role(const QAbstractItemModel* model, const QByteArray& name) {
    return role_for(model, name) >= 0;
}

int row_with_value(const QAbstractItemModel* model, int role, int value) {
    for (int row = 0; row < model->rowCount(); ++row) {
        if (model->data(model->index(row, 0), role).toInt() == value) return row;
    }
    return -1;
}

bool pump_until(QGuiApplication& app, const std::function<bool()>& done, int timeout_ms = 3000) {
    QElapsedTimer timer;
    timer.start();
    while (!done() && timer.elapsed() < timeout_ms) {
        app.processEvents();
        QThread::msleep(5);
    }
    return done();
}

}  // namespace

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    qputenv("DIAMOND_MCTS_SIMULATIONS", QByteArrayLiteral("1"));
#ifdef Q_OS_WIN
    if (qEnvironmentVariableIsEmpty("QT_MEDIA_BACKEND"))
        qputenv("QT_MEDIA_BACKEND", QByteArrayLiteral("windows"));
#endif
    QGuiApplication app(argc, argv);
    qInfo("controller contract: startup");
    NativeController controller;
    if (!require(controller.nativeRulesReady(), "native topology did not load")) return 1;
    const QMetaObject* meta = controller.metaObject();
    for (const char* property : {"phase", "currentPlayerId", "isCurrentPlayerAi",
             "statusMessage", "errorMessage", "winnerId", "winnerName", "resultSummary",
             "proposalHopCount", "lastMoveText"}) {
        if (!require(meta->indexOfProperty(property) >= 0, "Python controller property is missing")) return 1;
    }
    for (const char* signal : {"errorRaised(QString)", "gameFinished(int)", "playerFinished(int,int)"}) {
        if (!require(meta->indexOfSignal(signal) >= 0, "Python controller signal is missing")) return 1;
    }
    for (const char* method : {"newGame()", "requestAiMove()", "shutdown()"}) {
        if (!require(meta->indexOfMethod(method) >= 0, "Python controller method is missing")) return 1;
    }
    if (!require(controller.soundAvailable(), "native move sound is unavailable")) return 1;
    if (!require(controller.soundEnabled(), "native move sound does not start enabled")) return 1;
    if (!require(std::abs(controller.soundVolume() - 0.6) < 0.001,
                 "native move sound has the wrong default volume")) return 1;
    controller.setSoundEnabled(false);
    if (!require(!controller.soundEnabled() && controller.soundAvailable(),
                 "muting incorrectly unloads the move sound")) return 1;
    controller.setSoundVolume(0.35);
    if (!require(controller.soundEnabled() && std::abs(controller.soundVolume() - 0.35) < 0.001,
                 "setting a positive volume did not unmute or round-trip")) return 1;
    const int preview_requests = controller.soundPlayRequestCount();
    controller.previewSound();
    if (!require(controller.soundPlayRequestCount() == preview_requests + 1,
                 "sound preview did not request playback")) return 1;
    if (!require(pump_until(app, [&controller] {
            return controller.soundLoaded() || !controller.soundStatus().isEmpty();
        }, 5000) && controller.soundLoaded(),
        "Qt Multimedia could not load/decode the packaged move sound")) return 1;
    qInfo("controller contract: sound");
    controller.startMatch(QVariantList{1, 2}, QVariantList{});

    auto* geometry = qobject_cast<GeometryModel*>(controller.geometry());
    if (!require(geometry && geometry->holes().size() == soo::kBoardSize,
                 "native geometry does not expose all board holes")) return 1;
    const QVariantList camps = geometry->camps();
    if (!require(camps.size() == 6, "native geometry does not expose six camps")) return 1;
    for (const QVariant& value : camps) {
        const QVariantMap camp = value.toMap();
        if (!require(camp.value("points").toList().size() == 3 &&
                     camp.value("holes").toList().size() == soo::kCampSize &&
                     !camp.value("key").toString().isEmpty(),
                     "native camp geometry differs from the Python board")) return 1;
    }

    auto* players = qobject_cast<QAbstractItemModel*>(controller.playerModel());
    if (!require(players && players->rowCount() == 2, "player model does not contain every seat")) return 1;
    for (const QByteArray role : {"playerId", "name", "kindLabel", "color", "isCurrent",
             "isAi", "homeCount", "campSize", "hasFinished", "turnIndex", "place", "placeLabel"}) {
        if (!require(has_role(players, role), "player model role is missing")) return 1;
    }

    auto* board = qobject_cast<QAbstractItemModel*>(controller.boardModel());
    if (!require(board != nullptr, "board model is not an item model")) return 1;
    const int legal_step = role_for(board, QByteArrayLiteral("isLegalStep"));
    const int legal_jump = role_for(board, QByteArrayLiteral("isLegalJump"));
    if (!require(legal_step >= 0 && legal_jump >= 0, "legal destination roles are missing")) return 1;

    int source = -1;
    for (int position = 0; position < board->rowCount(); ++position) {
        controller.selectPosition(position);
        if (controller.selectedPosition() == position) {
            source = position;
            break;
        }
    }
    if (!require(source >= 0, "no current-player piece could be selected")) return 1;

    int destination = -1;
    for (int row = 0; row < board->rowCount(); ++row) {
        const QModelIndex index = board->index(row, 0);
        if (board->data(index, legal_step).toBool() || board->data(index, legal_jump).toBool()) {
            destination = row;
            break;
        }
    }
    if (!require(destination >= 0, "selected piece has no published legal destination")) return 1;

    const int turn_before = controller.turnNumber();
    controller.selectPosition(destination);

    if (!require(controller.turnNumber() == turn_before,
                 "choosing a destination committed before confirmation")) return 1;
    if (!require(controller.hasProposal(), "choosing a destination did not create a proposal")) return 1;
    if (!require(controller.canConfirm(), "proposal cannot be confirmed")) return 1;
    if (!require(controller.canCancel(), "proposal cannot be cancelled")) return 1;
    if (!require(controller.proposalPathIds().size() >= 2, "proposal has no canonical path")) return 1;

    auto* pieces = qobject_cast<QAbstractItemModel*>(controller.pieceModel());
    if (!require(pieces != nullptr, "piece model is not an item model")) return 1;
    const int piece_id_role = role_for(pieces, QByteArrayLiteral("pieceId"));
    const int position_id_role = role_for(pieces, QByteArrayLiteral("positionId"));
    const int source_piece_row = row_with_value(pieces, position_id_role, source);
    if (!require(source_piece_row >= 0, "source piece is missing before confirmation")) return 1;
    const int moving_piece_id = pieces->data(pieces->index(source_piece_row, 0), piece_id_role).toInt();

    const int move_sound_before = controller.soundPlayRequestCount();
    const int expected_hop_sounds = controller.proposalPathIds().size() - 1;
    controller.confirmProposal();
    if (!require(controller.turnNumber() == turn_before + 1, "confirmation did not commit the move")) return 1;
    if (!require(!controller.hasProposal(), "proposal survived confirmation")) return 1;
    if (!require(!controller.canSelect(), "board remained selectable during piece animation")) return 1;

    if (!require(pump_until(app, [&controller] { return controller.canSelect(); }),
                 "piece animation did not finish")) return 1;
    const int moved_row = row_with_value(pieces, piece_id_role, moving_piece_id);
    if (!require(moved_row >= 0, "moving piece identity was recreated")) return 1;
    if (!require(pieces->data(pieces->index(moved_row, 0), position_id_role).toInt() == destination,
                 "moving piece did not land at the confirmed destination")) return 1;
    if (!require(controller.soundPlayRequestCount() == move_sound_before + expected_hop_sounds,
                 "confirmed move did not request one sound per landing")) return 1;

    auto* history = qobject_cast<QAbstractItemModel*>(controller.historyModel());
    if (!require(history && history->rowCount() == 1, "confirmed move is missing from history")) return 1;
    if (!require(has_role(history, "playerId") && has_role(history, "isAi"),
                 "history parity roles are missing")) return 1;
    const int last_source_role = role_for(board, QByteArrayLiteral("isLastMoveSource"));
    const int last_dest_role = role_for(board, QByteArrayLiteral("isLastMoveDest"));
    if (!require(board->data(board->index(source, 0), last_source_role).toBool() &&
                 board->data(board->index(destination, 0), last_dest_role).toBool(),
                 "last-move board markers were not retained")) return 1;

    QTemporaryDir save_dir;
    const QUrl save_url = QUrl::fromLocalFile(save_dir.filePath(QStringLiteral("roundtrip.json")));
    if (!require(save_dir.isValid() && controller.saveGame(save_url), "native save failed")) return 1;
    NativeController loaded_controller;
    if (!require(loaded_controller.loadGame(save_url), "native load failed")) return 1;
    auto* loaded_history = qobject_cast<QAbstractItemModel*>(loaded_controller.historyModel());
    if (!require(loaded_controller.turnNumber() == controller.turnNumber() &&
                 loaded_history && loaded_history->rowCount() == history->rowCount() &&
                 loaded_controller.canUndo(), "save/load did not restore history and undo state")) return 1;

    controller.undoLastMove();
    if (!require(controller.turnNumber() == turn_before && history->rowCount() == 0,
                 "undo did not restore the prior state and history")) return 1;
    const QString old_game_label = controller.gameLabel();
    controller.newGame();
    if (!require(controller.turnNumber() == 1 && controller.gameLabel() != old_game_label,
                 "new game did not reset and increment the game label")) return 1;
    qInfo("controller contract: human controller");

    NativeController ai_controller;
    ai_controller.startMatch(QVariantList{1, 2}, QVariantList{2});
    auto* ai_board = qobject_cast<QAbstractItemModel*>(ai_controller.boardModel());
    const int ai_legal_step = role_for(ai_board, QByteArrayLiteral("isLegalStep"));
    const int ai_legal_jump = role_for(ai_board, QByteArrayLiteral("isLegalJump"));
    int human_source = -1;
    for (int position = 0; position < ai_board->rowCount(); ++position) {
        ai_controller.selectPosition(position);
        if (ai_controller.selectedPosition() == position) {
            human_source = position;
            break;
        }
    }
    if (!require(human_source >= 0, "AI match has no selectable human piece")) return 1;
    int human_destination = -1;
    for (int row = 0; row < ai_board->rowCount(); ++row) {
        const QModelIndex index = ai_board->index(row, 0);
        if (ai_board->data(index, ai_legal_step).toBool() || ai_board->data(index, ai_legal_jump).toBool()) {
            human_destination = row;
            break;
        }
    }
    if (!require(human_destination >= 0, "AI match human piece has no destination")) return 1;
    ai_controller.selectPosition(human_destination);
    ai_controller.confirmProposal();

    if (!require(pump_until(app, [&ai_controller] {
            return ai_controller.hasProposal() && ai_controller.proposalIsAi();
        }, 30000), "AI result was not published as a confirmable proposal")) return 1;
    const int ai_turn = ai_controller.turnNumber();
    const QString first_ai_move = ai_controller.proposalSummary();
    if (!require(ai_controller.canConfirm(), "AI proposal cannot be confirmed")) return 1;

    ai_controller.thinkAgain();
    if (!require(pump_until(app, [&ai_controller, &first_ai_move] {
            return ai_controller.hasProposal() && ai_controller.proposalIsAi()
                && ai_controller.proposalSummary() != first_ai_move;
        }, 30000), "Think Again did not produce a different proposal")) return 1;
    if (!require(ai_controller.turnNumber() == ai_turn,
                 "Think Again changed authoritative game state")) return 1;

    ai_controller.confirmProposal();
    if (!require(pump_until(app, [&ai_controller, ai_turn] {
            return ai_controller.turnNumber() == ai_turn + 1 && ai_controller.canSelect();
        }, 5000), "confirming the AI proposal did not finish its move")) return 1;
    qInfo("controller contract: AI controller");

    NativeController terminal_controller;
    terminal_controller.startMatch(QVariantList{1, 2}, QVariantList{});
    const auto& target = soo::topology().camp_positions[terminal_controller.playerCount() == 2 ? 5 : 5];
    int final_hole = target.back();
    int entry_hole = -1;
    for (int direction = 0; direction < soo::kDirections; ++direction) {
        const int candidate = soo::topology().neighbour[final_hole][direction];
        if (candidate < 0) continue;
        bool inside_target = false;
        for (uint8_t target_hole : target) if (candidate == target_hole) inside_target = true;
        if (!inside_target) { entry_hole = candidate; break; }
    }
    if (!require(entry_hole >= 0, "could not construct terminal controller fixture")) return 1;
    QJsonArray terminal_occupancy;
    for (int position = 0; position < soo::kBoardSize; ++position) {
        int occupant = position == entry_hole ? 1 : 0;
        for (uint8_t target_hole : target)
            if (position == target_hole && position != final_hole) occupant = 1;
        terminal_occupancy.push_back(occupant);
    }
    QTemporaryDir terminal_dir;
    const QString terminal_path = terminal_dir.filePath(QStringLiteral("terminal-v1.json"));
    QFile terminal_file(terminal_path);
    if (!terminal_file.open(QIODevice::WriteOnly)) return 1;
    terminal_file.write(QJsonDocument(QJsonObject{{"version", 1},
        {"order", QJsonArray{1, 2}}, {"aiSeats", QJsonArray{}},
        {"occupancy", terminal_occupancy}, {"currentPlayer", 1},
        {"turnNumber", 40}}).toJson());
    terminal_file.close();
    if (!require(terminal_controller.loadGame(QUrl::fromLocalFile(terminal_path)),
                 "terminal fixture did not load")) return 1;
    int finished_winner = 0;
    QObject::connect(&terminal_controller, &NativeController::gameFinished,
                     [&finished_winner](int winner) { finished_winner = winner; });
    terminal_controller.selectPosition(entry_hole);
    terminal_controller.selectPosition(final_hole);
    terminal_controller.confirmProposal();
    if (!require(pump_until(app, [&finished_winner] { return finished_winner != 0; }),
                 "terminal move did not emit gameFinished")) return 1;
    if (!require(terminal_controller.isGameOver() && terminal_controller.winnerId() == 1 &&
                 terminal_controller.standings().size() == 2 && !terminal_controller.canSelect(),
                 "terminal controller state does not match the Python oracle")) return 1;

    qInfo("controller contract: complete");

    return 0;
}
