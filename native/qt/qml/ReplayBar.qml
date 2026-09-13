import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

Rectangle {
    id: root
    required property var controller
    objectName: "replayBar"
    implicitHeight: 106
    color: Theme.surface
    radius: Theme.radiusSmall
    border.color: Theme.border
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 6
        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "REPLAY  ·  " + (root.controller.replayIndex === 0 ? "Starting position" : "Move " + root.controller.replayIndex + " of " + root.controller.replayCount)
                color: Theme.text
                font.family: Theme.fontFamily
                font.weight: Theme.weightBold
                font.pixelSize: Theme.fontSmall
                Layout.fillWidth: true
            }
            Text { text: "Read-only · live game preserved"; color: Theme.textMuted; font.pixelSize: Theme.fontSmall }
            ActionButton { text: "Return to game"; onClicked: root.controller.leaveReplay() }
        }
        RowLayout {
            Layout.fillWidth: true
            ActionButton { text: "First"; enabled: root.controller.replayIndex > 0; onClicked: root.controller.seekReplay(0); Accessible.name: "First position" }
            ActionButton { text: "Back"; enabled: root.controller.replayIndex > 0; onClicked: root.controller.seekReplay(root.controller.replayIndex - 1); Accessible.name: "Previous move" }
            ActionButton { text: root.controller.replayPlaying ? "Pause" : "Play"; primary: true; onClicked: root.controller.toggleReplayPlayback() }
            ActionButton { text: "Next"; enabled: root.controller.replayIndex < root.controller.replayCount; onClicked: root.controller.seekReplay(root.controller.replayIndex + 1); Accessible.name: "Next move" }
            Slider {
                Layout.fillWidth: true
                from: 0
                to: Math.max(1, root.controller.replayCount)
                stepSize: 1
                snapMode: Slider.SnapAlways
                value: root.controller.replayIndex
                onMoved: root.controller.seekReplay(Math.round(value))
                Accessible.name: "Replay move"
            }
            ActionButton { text: "Last"; onClicked: root.controller.seekReplay(root.controller.replayCount); Accessible.name: "Latest position" }
        }
    }
}
