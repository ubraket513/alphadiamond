import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Dialogs
import QtQuick.Layouts
import Style

ApplicationWindow {
    id: window

    width: 1440
    height: 900
    minimumWidth: 980
    minimumHeight: 640
    visible: true
    title: "Diamond — Controller Console"
    color: Theme.background

    // `controller` is injected as a context property from Python.
    readonly property var ctrl: controller

    // -- header ----------------------------------------------------------
    header: Rectangle {
        implicitHeight: 60
        color: Theme.surface

        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: Theme.border
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: Theme.spacingLarge
            anchors.rightMargin: Theme.spacingLarge

            Text {
                text: "Diamond AI"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontTitle
            }
            Text {
                Layout.fillWidth: true
                text: "  ·  " + window.ctrl.playerCount + "-player operator console"
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                elide: Text.ElideRight
            }
            Text {
                text: window.ctrl.gameLabel
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
            }
        }
    }

    // -- body ------------------------------------------------------------
    RowLayout {
        anchors.fill: parent
        anchors.margins: Theme.spacingLarge
        spacing: Theme.spacingLarge

        Board {
            id: board
            controller: window.ctrl
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumWidth: 420
        }

        SidePanel {
            controller: window.ctrl
            Layout.preferredWidth: Theme.panelWidth
            Layout.maximumWidth: Theme.panelWidth
            Layout.fillHeight: true
        }
    }

    footer: StatusBar {
        controller: window.ctrl
        onNewGameRequested: newGameDialog.open()
        onSaveRequested: saveDialog.open()
        onLoadRequested: loadDialog.open()
    }

    // -- keyboard shortcuts ----------------------------------------------
    // Each one re-checks controller state, so no shortcut can confirm or undo
    // something the current phase does not allow.
    Shortcut {
        sequence: "Return"
        enabled: window.ctrl.canConfirm
        onActivated: window.ctrl.confirmProposal()
    }
    Shortcut {
        sequence: "Enter"
        enabled: window.ctrl.canConfirm
        onActivated: window.ctrl.confirmProposal()
    }
    Shortcut {
        sequence: "Escape"
        enabled: window.ctrl.canCancel
        onActivated: window.ctrl.cancelProposal()
    }
    Shortcut {
        sequences: [StandardKey.Undo]
        enabled: window.ctrl.canUndo
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
