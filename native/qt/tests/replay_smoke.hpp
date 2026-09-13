#pragma once
#include <QElapsedTimer>
#include <QThread>
#include <QPointer>
#include "soo/rules.hpp"

struct ReplayTestAccess {
    static QVector<int> fixture(NativeController& c) {
        c.cancelSearch();
        c.shutting_down_ = true; // deterministic animation test, no inference
        c.ai_seats_.clear();
        c.state_history_.clear();
        c.history_.clear();
        c.clearProposal();
        c.selected_position_ = -1;
        c.legal_actions_.clear();
        const auto& n = soo::topology().neighbour;
        for (int source = 0; source < soo::kBoardSize; ++source)
            for (int d = 0; d < soo::kDirections; ++d) {
                int over = n[source][d];
                if (over < 0)
                    continue;
                int mid = n[over][d];
                if (mid < 0)
                    continue;
                for (int e = 0; e < soo::kDirections; ++e) {
                    if (d == e)
                        continue;
                    int over2 = n[mid][e];
                    if (over2 < 0)
                        continue;
                    int dest = n[over2][e];
                    if (dest < 0 || dest == source || over2 == source || dest == over)
                        continue;
                    soo::State state{};
                    state.current_player = 1;
                    state.occupancy[source] = 1;
                    state.occupancy[over] = 2;
                    state.occupancy[over2] = 2;
                    std::vector<uint8_t> path;
                    if (!soo::canonical_move_path(state, source, dest, path) || path.size() != 3)
                        continue;
                    c.state_ = state;
                    c.piece_model_->setRows({});
                    c.refreshModels();
                    return {path[0], path[1], path[2]};
                }
            }
        return {};
    }
};

inline int replaySmoke(QGuiApplication& app, NativeController& controller, QObject* root,
                       const QString& screenshot) {
    auto require = [](bool value, const char* why) {
        if (!value)
            qCritical("replay smoke: %s", why);
        return value;
    };
    auto pump = [&](int ms) {
        QElapsedTimer t;
        t.start();
        while (t.elapsed() < ms) {
            app.processEvents();
            QThread::msleep(2);
        }
    };
    const auto path = ReplayTestAccess::fixture(controller);
    if (!require(path.size() == 3, "no turning two-jump fixture"))
        return 1;
    pump(50);
    auto* window = qobject_cast<QQuickWindow*>(root);
    auto findPiece = [&](auto&& self, QQuickItem* item) -> QQuickItem* {
        if (item->objectName().startsWith("boardPiece-") &&
            item->property("positionId").toInt() == path.front())
            return item;
        for (auto* child : item->childItems())
            if (auto* found = self(self, child))
                return found;
        return nullptr;
    };
    QPointer<QQuickItem> piece = findPiece(findPiece, window->contentItem());
    if (!require(piece, "piece delegate missing"))
        return 1;
    controller.selectPosition(path.front());
    controller.selectPosition(path.back());
    const int sounds = controller.soundPlayRequestCount();
    controller.confirmProposal();
    auto* geometry = qobject_cast<GeometryModel*>(controller.geometry());
    for (int hop = 1; hop < path.size(); ++hop) {
        pump(hop == 1 ? 570 : 300);
        const auto point = geometry->holes()[path[hop]].toMap();
        if (!require(piece &&
                         std::abs(piece->property("animUnitX").toDouble() - point["x"].toDouble()) <
                             .01 &&
                         std::abs(piece->property("animUnitY").toDouble() - point["y"].toDouble()) <
                             .01,
                     "rendered piece skipped a turning landing"))
            return 1;
    }
    pump(150);
    if (!require(controller.soundPlayRequestCount() == sounds + 2, "expected one SFX per landing"))
        return 1;
    const int turn = controller.turnNumber();
    controller.seekReplay(0);
    if (!require(controller.replayActive() && !controller.canSelect() && !controller.canUndo() &&
                     !controller.canConfirm(),
                 "replay permits live actions"))
        return 1;
    controller.confirmProposal();
    controller.undoLastMove();
    if (!require(controller.turnNumber() == turn && controller.replayCount() == 1,
                 "replay mutated live history"))
        return 1;
    controller.toggleReplayPlayback();
    pump(2300);
    if (!require(controller.replayIndex() == 1 && controller.soundPlayRequestCount() == sounds + 4,
                 "replay did not play both jumps"))
        return 1;
    if (!screenshot.isEmpty())
        window->grabWindow().save(screenshot);
    if (!screenshot.isEmpty()) {
        auto* drawer = root->findChild<QObject*>(QStringLiteral("historyDrawer"));
        drawer->setProperty("open", true);
        pump(500);
        window->grabWindow().save(screenshot + QStringLiteral(".history.png"));
        drawer->setProperty("open", false);
    }
    controller.leaveReplay();
    if (!require(!controller.replayActive() && controller.turnNumber() == turn &&
                     controller.canSelect(),
                 "live state was not restored"))
        return 1;
    qInfo("replay smoke: turning route, rendered landings, per-hop sound, playback, live-state "
          "isolation passed");
    return 0;
}
