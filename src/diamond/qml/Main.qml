import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import QtQuick.Layouts
import QtQuick.Window
import Style

ApplicationWindow {
    id: window

    width: 1440
    height: 900
    minimumWidth: 980
    minimumHeight: 640
    visible: true
    title: "Diamond — Controller Console"
    color: "transparent"

    // Frameless: TitleBar.qml provides the chrome the OS would normally draw.
    // Qt.Window keeps the taskbar entry that a bare FramelessWindowHint loses.
    flags: Qt.Window | Qt.FramelessWindowHint

    // `controller` is injected as a context property from Python.
    readonly property var ctrl: controller

    // Qt shortcuts are application-wide and outrank an item's `Keys` handlers,
    // so a modal dialog would never see Return/Esc unless the board's own
    // shortcuts stand down while one is open.
    readonly property bool dialogOpen:
        newGameDialog.visible || gameOverDialog.visible || placeDialog.visible
        || aboutDialog.visible || soundDialog.visible
        || saveDialog.visible || loadDialog.visible

    // A frameless window paints its own surround. `color: "transparent"` above
    // lets this rounded/bordered rectangle define the window edge instead of a
    // hard system rectangle.
    background: Rectangle {
        color: Theme.background
        border.width: 1
        border.color: window.active ? Theme.borderStrong : Theme.border
    }

    // -- window chrome ---------------------------------------------------
    header: TitleBar {
        id: titleBar
        controller: window.ctrl
        window: window
        onNewGameRequested: newGameDialog.open()
        onSaveRequested: saveDialog.open()
        onLoadRequested: loadDialog.open()
        onSoundsRequested: soundDialog.open()
        onAboutRequested: aboutDialog.open()
    }

    // -- body ------------------------------------------------------------
    RowLayout {
        anchors.fill: parent
        anchors.margins: Theme.spacingLarge
        spacing: Theme.spacingLarge

        // Collapsing animates the laid-out width, so the board grows into the
        // space in the same frames rather than snapping afterwards. `visible`
        // only drops once the panel is actually gone, which keeps it out of
        // hit-testing and focus while leaving it on screen for the animation.
        SidePanel {
            id: sidePanel
            controller: window.ctrl
            clip: true
            visible: Layout.preferredWidth > 0.5
            opacity: titleBar.panelVisible ? 1 : 0
            Layout.preferredWidth: titleBar.panelVisible ? Theme.panelWidth : 0
            Layout.maximumWidth: Theme.panelWidth
            Layout.fillHeight: true

            Behavior on Layout.preferredWidth {
                NumberAnimation {
                    duration: Theme.panelDuration
                    easing.type: Easing.Bezier
                    easing.bezierCurve: Theme.easeEmphasized
                }
            }
            // Fades a little faster than it slides, so the content is gone
            // before the panel is narrow enough to squash it.
            Behavior on opacity {
                NumberAnimation {
                    duration: Math.round(Theme.panelDuration * 0.6)
                    easing.type: Easing.Bezier
                    easing.bezierCurve: Theme.easeStandard
                }
            }
        }

        Board {
            id: board
            controller: window.ctrl
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumWidth: 420
        }
    }

    // -- keyboard shortcuts ----------------------------------------------
    // Each one re-checks controller state, so no shortcut can confirm or undo
    // something the current phase does not allow.
    Shortcut {
        sequence: "Return"
        enabled: !window.dialogOpen && window.ctrl.canConfirm
        onActivated: window.ctrl.confirmProposal()
    }
    Shortcut {
        sequence: "Enter"
        enabled: !window.dialogOpen && window.ctrl.canConfirm
        onActivated: window.ctrl.confirmProposal()
    }
    Shortcut {
        sequence: "Escape"
        enabled: !window.dialogOpen && window.ctrl.canCancel
        onActivated: window.ctrl.cancelProposal()
    }
    Shortcut {
        sequences: [StandardKey.Undo]
        enabled: !window.dialogOpen && window.ctrl.canUndo
        onActivated: window.ctrl.undoLastMove()
    }
    Shortcut {
        sequences: [StandardKey.Save]
        onActivated: saveDialog.open()
    }
    Shortcut {
        sequences: [StandardKey.New]
        onActivated: newGameDialog.open()
    }
    Shortcut {
        // Advertised as Ctrl+O in the File menu, so it has to exist.
        sequences: [StandardKey.Open]
        onActivated: loadDialog.open()
    }

    // -- resize edges ------------------------------------------------------
    // A frameless window loses the OS resize border, so the eight edges and
    // corners are re-created here. Each hands off to startSystemResize, which
    // keeps native behaviour (aero snap, min/max sizes, live update).
    Item {
        anchors.fill: parent
        z: 9999

        Repeater {
            model: [
                { e: Qt.TopEdge,                     cx: 0,  cy: -1, cur: Qt.SizeVerCursor },
                { e: Qt.BottomEdge,                  cx: 0,  cy: 1,  cur: Qt.SizeVerCursor },
                { e: Qt.LeftEdge,                    cx: -1, cy: 0,  cur: Qt.SizeHorCursor },
                { e: Qt.RightEdge,                   cx: 1,  cy: 0,  cur: Qt.SizeHorCursor },
                { e: Qt.TopEdge | Qt.LeftEdge,       cx: -1, cy: -1, cur: Qt.SizeFDiagCursor },
                { e: Qt.TopEdge | Qt.RightEdge,      cx: 1,  cy: -1, cur: Qt.SizeBDiagCursor },
                { e: Qt.BottomEdge | Qt.LeftEdge,    cx: -1, cy: 1,  cur: Qt.SizeBDiagCursor },
                { e: Qt.BottomEdge | Qt.RightEdge,   cx: 1,  cy: 1,  cur: Qt.SizeFDiagCursor }
            ]

            delegate: Item {
                required property var modelData

                readonly property real m: Theme.resizeMargin
                // Corners (cx and cy both set) are square; edges span a side.
                readonly property bool corner: modelData.cx !== 0 && modelData.cy !== 0

                width: (modelData.cx === 0) ? parent.width - 2 * m : m
                height: (modelData.cy === 0) ? parent.height - 2 * m : m
                x: modelData.cx < 0 ? 0 : (modelData.cx > 0 ? parent.width - m : m)
                y: modelData.cy < 0 ? 0 : (modelData.cy > 0 ? parent.height - m : m)

                // Resizing is meaningless while maximised.
                enabled: window.visibility === Window.Windowed

                HoverHandler {
                    enabled: parent.enabled
                    cursorShape: modelData.cur
                }
                DragHandler {
                    target: null
                    enabled: parent.enabled
                    grabPermissions: PointerHandler.CanTakeOverFromAnything
                    onActiveChanged: if (active) window.startSystemResize(modelData.e)
                }
            }
        }
    }

    // -- dialogs ----------------------------------------------------------
    NewMatchDialog {
        id: newGameDialog
        objectName: "newMatchDialog"
        onConfirmed: (order, aiSeats) => window.ctrl.startMatch(order, aiSeats)
    }

    FileDialog {
        id: saveDialog
        title: "Save game"
        fileMode: FileDialog.SaveFile
        defaultSuffix: "json"
        nameFilters: ["Diamond save (*.json)"]
        currentFolder: window.ctrl.defaultSaveDir
        onAccepted: window.ctrl.saveGame(selectedFile)
    }

    FileDialog {
        id: loadDialog
        title: "Load game"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Diamond save (*.json)"]
        currentFolder: window.ctrl.defaultSaveDir
        onAccepted: window.ctrl.loadGame(selectedFile)
    }

    ResultDialog {
        id: gameOverDialog
        objectName: "resultDialog"
        onRematchRequested: newGameDialog.open()
    }

    // A seat finishing is worth announcing on its own in a 3-player match,
    // where play carries on afterwards to settle the remaining places.
    AppDialog {
        id: placeDialog
        objectName: "placeDialog"

        property string playerName: ""
        property string placeLabel: ""

        title: playerName + " finished " + placeLabel
        message: "All ten pieces are home. " + playerName
                 + " sits out the rest of the match while the remaining places are decided."
        acceptText: "Continue"
        showReject: false
    }

    SoundDialog {
        id: soundDialog
        objectName: "soundDialog"
        controller: window.ctrl
    }

    AppDialog {
        id: aboutDialog
        objectName: "aboutDialog"
        title: "Diamond"
        message: "Controller console for running a real 2- or 3-player Diamond match.

"
                 + "The board is a 73-hole star; each camp is the 10-hole triangle formed by a "
                 + "star point and the hexagon side it stands on."
        acceptText: "Close"
        showReject: false
    }

    Connections {
        target: window.ctrl

        function onGameFinished(winnerId) {
            placeDialog.close()
            gameOverDialog.open()
        }

        function onPlayerFinished(playerId, place) {
            // The final placement is covered by the results dialog instead.
            if (window.ctrl.isGameOver)
                return
            var rows = window.ctrl.standings
            for (var i = 0; i < rows.length; i++) {
                if (rows[i].playerId === playerId) {
                    placeDialog.playerName = rows[i].name
                    placeDialog.placeLabel = rows[i].placeLabel
                    placeDialog.open()
                    return
                }
            }
        }
    }
}
