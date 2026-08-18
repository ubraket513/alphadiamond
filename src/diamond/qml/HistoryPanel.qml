import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    required property var controller
    title: "MOVE HISTORY"

    ListView {
        id: list
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.minimumHeight: 40
        clip: true
        model: root.controller.historyModel
        spacing: 2
        boundsBehavior: Flickable.StopAtBounds

        onCountChanged: positionViewAtEnd()

        ScrollBar.vertical: PanelScrollBar {}

        delegate: Rectangle {
            id: entry

            required property int    turnNumber
            required property string playerLabel
            required property string playerColor
            required property string moveText
            required property string pathText
            required property int    hopCount

            width: list.width
            height: column.implicitHeight + 8
            color: expanded ? Theme.surfaceAlt : "transparent"
            radius: Theme.radiusSmall

            property bool expanded: false

            ColumnLayout {
                id: column
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.margins: 4
                spacing: 2

                RowLayout {
                    Layout.fillWidth: true
                    spacing: Theme.spacing

                    Text {
                        text: entry.turnNumber + "."
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontSmall
                        Layout.preferredWidth: 30
                        horizontalAlignment: Text.AlignRight
                    }
                    Rectangle {
                        width: 8; height: 8; radius: 4
                        color: entry.playerColor
                        border.width: 1
                        border.color: Theme.lattice
                    }
                    Text {
                        text: entry.playerLabel
                        color: Theme.textMuted
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontSmall
                        Layout.preferredWidth: 24
                    }
                    Text {
                        Layout.fillWidth: true
                        text: entry.moveText
                        color: Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontSmall
                    }
                    Text {
                        visible: entry.hopCount > 1
                        text: entry.hopCount + " hops"
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontTiny
                    }
                }

                Text {
                    Layout.fillWidth: true
                    visible: entry.expanded && entry.hopCount > 1
                    text: entry.pathText
                    color: Theme.textMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                    wrapMode: Text.WordWrap
                    leftPadding: 62
                }
            }

            MouseArea {
                anchors.fill: parent
                enabled: entry.hopCount > 1
                cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                onClicked: entry.expanded = !entry.expanded
            }
        }

        Text {
            anchors.centerIn: parent
            visible: list.count === 0
            text: "No moves yet"
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
    }
}
