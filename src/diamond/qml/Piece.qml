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

    // Selection ring, in the UI accent rather than the piece's own colour: it
    // marks interface state, not identity, so it must read the same on every
    // player's pieces.
    Rectangle {
        anchors.centerIn: parent
        visible: root.highlighted
        width: root.unitScale * Theme.pieceRatio * 2 + root.unitScale * 0.26
        height: width
        radius: width / 3
        color: "transparent"
        border.width: Math.max(2, root.unitScale * 0.08)
        border.color: Theme.accent
        antialiasing: true
    }

    Rectangle {
        anchors.centerIn: parent
        width: root.unitScale * Theme.pieceRatio * 2
        height: width
        radius: width / 2
        color: root.pieceColor
        // The rim is the piece's own colour, darkened — it defines the edge
        // without adding a second hue, and works for red, yellow and green
        // alike where one fixed grey would not.
        border.width: Math.max(1, root.unitScale * 0.035)
        border.color: Qt.darker(root.pieceColor, 1.3)
        // Pieces that cannot move this turn recede rather than disappear.
        opacity: root.active ? 1.0 : 0.72
        antialiasing: true

        Behavior on opacity { NumberAnimation { duration: Theme.fadeDuration } }
    }
}
