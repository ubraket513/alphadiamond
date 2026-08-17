import QtQuick
import QtQuick.Layouts
import Style

// Footer: last committed move on the left, controller actions on the right.
Rectangle {
    id: root

    required property var controller

    signal newGameRequested()
    signal saveRequested()
    signal loadRequested()

    implicitHeight: 56
    color: Theme.surface
    border.width: 0

    Rectangle {
        anchors.top: parent.top
        width: parent.width
        height: 1
        color: Theme.border
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: Theme.spacingLarge
        anchors.rightMargin: Theme.spacingLarge
        spacing: Theme.spacing

        Text {
            text: "Last move"
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        Text {
            text: root.controller.lastMoveText
            color: Theme.text
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontBody
        }

        Rectangle {
            Layout.preferredWidth: 1
            Layout.preferredHeight: 20
            color: Theme.border
        }

        Text {
            Layout.fillWidth: true
            text: root.controller.errorMessage !== ""
                  ? root.controller.errorMessage
                  : root.controller.statusMessage
            color: root.controller.errorMessage !== "" ? Theme.danger : Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            elide: Text.ElideRight
        }

        ActionButton {
            text: "Undo  (Ctrl+Z)"
            enabled: root.controller.canUndo
            onClicked: root.controller.undoLastMove()
        }
        ActionButton {
            text: "Save  (Ctrl+S)"
            onClicked: root.saveRequested()
        }
        ActionButton {
            text: "Load"
            onClicked: root.loadRequested()
        }
        ActionButton {
            text: "New Game  (Ctrl+N)"
            onClicked: root.newGameRequested()
        }
    }
}
