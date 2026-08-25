import QtQuick
import Style

Item {
    id: root
    objectName: "historyDrawer"
    required property var controller
    property bool open: false

    visible: open || panel.x < width
    enabled: visible

    Rectangle {
        anchors.fill: parent
        color: Theme.shadowSoft
        opacity: root.open ? 1 : 0
        Behavior on opacity { NumberAnimation { duration: Theme.durationBase } }
        TapHandler { onTapped: root.open = false }
    }

    Rectangle {
        id: panel
        width: Math.min(390, root.width * 0.42)
        height: root.height
        x: root.open ? root.width - width : root.width
        color: Theme.surface
        border.width: 1
        border.color: Theme.border

        Behavior on x {
            NumberAnimation {
                duration: Theme.panelDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeEmphasized
            }
        }

        Text {
            id: heading
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.margins: Theme.spacingLarge
            text: "History"
            color: Theme.text
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontLarge
            font.weight: Theme.weightBold
        }

        ActionButton {
            id: closeButton
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: Theme.spacing
            width: 74
            text: "Close"
            onClicked: root.open = false
        }

        HistoryPanel {
            objectName: "drawerHistoryPanel"
            controller: root.controller
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: heading.bottom
            anchors.bottom: parent.bottom
            anchors.margins: Theme.spacingLarge
            anchors.topMargin: Theme.spacing
        }
    }
}

