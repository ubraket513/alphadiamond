import QtQuick
import Style

// A player's piece. Its position is driven by the controller one lattice hop at
// a time; the Behaviors turn that into a smooth multi-hop animation without the
// game state ever knowing about animation.
Item {
    id: root

    required property int    playerId
    required property color  pieceColor
    property real  unitScale: 1
    property bool  highlighted: false
    property bool  active: false      // belongs to the player to move

    Behavior on x { NumberAnimation { duration: Theme.hopDuration; easing.type: Easing.InOutQuad } }
    Behavior on y { NumberAnimation { duration: Theme.hopDuration; easing.type: Easing.InOutQuad } }

    Rectangle {
        anchors.centerIn: parent
        width: root.unitScale * Theme.pieceRatio * 2
        height: width
        radius: width / 2
        color: root.pieceColor
        border.width: Math.max(1, root.unitScale * (root.active ? 0.07 : 0.03))
        border.color: Theme.lattice
        opacity: root.active ? 1.0 : 0.88
        antialiasing: true

        Rectangle {
            anchors.centerIn: parent
            visible: root.highlighted
            width: parent.width * 0.42
            height: width
            radius: width / 2
            color: Theme.lattice
            opacity: 0.65
            antialiasing: true
        }
    }
}
