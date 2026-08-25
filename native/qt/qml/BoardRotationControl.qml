import QtQuick
import QtQuick.Controls.Basic
import Style

Item {
    id: root
    objectName: "boardRotationControl"

    readonly property alias angle: dial.value

    width: 76
    height: 94

    function resetRotation() {
        dial.value = 0
        dial.forceActiveFocus()
    }

    Rectangle {
        anchors.fill: surface
        anchors.topMargin: 3
        radius: surface.radius
        color: Theme.shadowSoft
    }

    Rectangle {
        id: surface
        anchors.fill: parent
        radius: Theme.radiusLarge
        color: Theme.surface
        border.width: 1
        border.color: Theme.border
    }

    Dial {
        id: dial
        objectName: "boardRotationDial"

        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 8
        width: 58
        height: 58

        from: -180
        to: 180
        value: 0
        stepSize: 1
        snapMode: Dial.NoSnap
        wrap: true
        live: true
        startAngle: -180
        endAngle: 180
        wheelEnabled: true
        focusPolicy: Qt.StrongFocus

        Accessible.name: "Rotate board"
        Accessible.description: "Drag clockwise or counterclockwise to rotate the board continuously"

        background: Rectangle {
            x: 3
            y: 3
            width: dial.width - 6
            height: width
            radius: width / 2
            color: Theme.surfaceAlt
            border.width: dial.activeFocus ? 2 : 1
            border.color: dial.activeFocus ? Theme.accent : Theme.borderStrong
            antialiasing: true
        }

        handle: Item {}

        contentItem: Item {
            rotation: dial.angle

            Rectangle {
                anchors.top: parent.top
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.topMargin: 8
                width: 3
                height: 17
                radius: width / 2
                color: Theme.accent
                antialiasing: true
            }

            Rectangle {
                anchors.centerIn: parent
                width: 8
                height: 8
                radius: 4
                color: Theme.accent
                antialiasing: true
            }
        }

        HoverHandler { id: dialHover }
        ToolTip.visible: dialHover.hovered
        ToolTip.text: "Rotate board"
        ToolTip.delay: 500
    }

    Text {
        id: angleLabel
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        text: Math.round((dial.value + 360) % 360) + "°"
        color: labelHover.hovered ? Theme.accent : Theme.textMuted
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontSmall
        font.weight: Theme.weightMedium

        Accessible.role: Accessible.Button
        Accessible.name: "Reset board rotation"

        HoverHandler {
            id: labelHover
            cursorShape: Qt.PointingHandCursor
        }
        TapHandler { onTapped: root.resetRotation() }
    }
}
