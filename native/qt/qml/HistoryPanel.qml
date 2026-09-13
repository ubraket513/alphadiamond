import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

ColumnLayout {
    id: root
    required property var controller
    signal replayRequested()
    spacing: 10
    RowLayout {
        Layout.fillWidth: true
        Text {
            Layout.fillWidth: true
            text: root.controller.replayCount + " moves"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        ActionButton {
            text: "Replay from start"
            enabled: root.controller.replayCount > 0
            onClicked: { root.controller.seekReplay(0); root.replayRequested() }
        }
    }
    ListView {
        id: list
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        spacing: 6
        model: root.controller.historyModel
        boundsBehavior: Flickable.StopAtBounds
        ScrollBar.vertical: PanelScrollBar {}
        onCountChanged: positionViewAtEnd()
        delegate: Rectangle {
            id: entry
            required property int index
            required property int turnNumber
            required property string playerLabel
            required property string playerColor
            required property string moveText
            required property string pathText
            required property int hopCount
            width: list.width - 10
            height: content.implicitHeight + 24
            radius: Theme.radiusSmall
            color: root.controller.replayActive && root.controller.replayIndex === index + 1
                 ? Theme.surfaceAlt : Theme.surface
            border.width: 1
            border.color: root.controller.replayActive && root.controller.replayIndex === index + 1
                        ? Theme.selection : Theme.border
            ColumnLayout {
                id: content
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.margins: 12
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    Rectangle { width: 8; height: 8; radius: 4; color: entry.playerColor }
                    Text {
                        Layout.fillWidth: true
                        text: entry.playerLabel
                        elide: Text.ElideRight
                        color: Theme.textMuted
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontSmall
                    }
                    Text { text: "#" + entry.turnNumber; color: Theme.textFaint; font.pixelSize: Theme.fontTiny }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        Layout.fillWidth: true
                        text: entry.moveText
                        elide: Text.ElideRight
                        color: Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: Theme.weightBold
                    }
                    Text { text: entry.hopCount > 1 ? entry.hopCount + " jumps" : "1 move"; color: Theme.textMuted; font.pixelSize: Theme.fontTiny }
                }
                Text {
                    Layout.fillWidth: true
                    visible: entry.hopCount > 1
                    text: entry.pathText
                    wrapMode: Text.Wrap
                    color: Theme.textMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                }
            }
            TapHandler { onTapped: { root.controller.seekReplay(entry.index + 1); root.replayRequested() } }
            Accessible.role: Accessible.Button
            activeFocusOnTab: true
            Keys.onReturnPressed: { root.controller.seekReplay(index + 1); root.replayRequested() }
            Accessible.onPressAction: { root.controller.seekReplay(index + 1); root.replayRequested() }
            Accessible.name: "Replay move " + turnNumber + ", " + playerLabel + ", " + pathText
        }
        Text {
            anchors.centerIn: parent
            visible: list.count === 0
            text: "Your moves will appear here."
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
    }
}
