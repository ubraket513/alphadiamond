import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    required property var controller
    title: "GAME"

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

    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 1
        color: Theme.border
    }

    Text {
        Layout.fillWidth: true
        text: root.controller.isGameOver
              ? ("GAME OVER — " + root.controller.winnerName + " wins")
              : root.controller.phase.replace(/_/g, " ")
        color: root.controller.isGameOver ? Theme.danger : Theme.textMuted
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontSmall
        font.bold: root.controller.isGameOver
        elide: Text.ElideRight
    }
}
