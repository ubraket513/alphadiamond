import QtQuick
import QtQuick.Layouts
import Style

// Whose turn it is, and nothing else.
//
// Phase, the proposed move and the status/error line were all removed from
// here deliberately: the board itself shows the proposal (highlighted path,
// numbered hop markers), and this panel is meant to answer one question at a
// glance rather than be a log.
PanelSection {
    id: root
    required property var controller
    title: "Game"

    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingLarge

        ColumnLayout {
            spacing: 2
            Text {
                text: "Turn"
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
            }
            Text {
                text: root.controller.turnNumber
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontHuge
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2
            Text {
                text: "Current player"
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
            }
            RowLayout {
                spacing: 8
                Rectangle {
                    width: 12; height: 12; radius: 6
                    color: root.controller.currentPlayerColor
                    border.width: 1
                    border.color: Theme.lattice
                }
                Text {
                    Layout.fillWidth: true
                    text: root.controller.isGameOver
                          ? "—" : root.controller.currentPlayerName
                    color: Theme.text
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontLarge
                    elide: Text.ElideRight
                }
            }
        }
    }
}
