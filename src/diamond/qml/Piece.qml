import QtQuick
import Style

// A player's piece.
//
// The hop animation runs in *lattice* space, not screen space. Animating x/y
// directly conflates two different kinds of movement: a piece hopping to a new
// hole (which should ease over `hopDuration`) and the board being rescaled
// because the window or the side panel changed width (which should be
// instant). With a Behavior on x/y the second case gets interpolated too, so
// the pieces lag behind the lattice they are standing on and the whole board
// looks like it is sloshing.
//
// Interpolating `animUnit*` instead means a resize only changes `originX` /
// `unitScale`, which feed straight through to x/y with no Behavior attached —
// the piece tracks the board exactly, frame for frame — while a hop still
// eases, because that is the only thing that moves the lattice position.
Item {
    id: root

    required property int    playerId
    required property color  pieceColor
    required property real   unitX
    required property real   unitY

    // Board-to-screen mapping, supplied by Board.qml.
    property real originX: 0
    property real originY: 0
    property real unitScale: 1

    property bool highlighted: false   // this piece is the current selection

    // Lattice position, eased. The binding re-evaluates when the model moves
    // the piece, and the Behavior turns that step into a hop.
    property real animUnitX: unitX
    property real animUnitY: unitY

    Behavior on animUnitX {
        NumberAnimation {
            duration: Theme.hopDuration
            easing.type: Easing.Bezier
            easing.bezierCurve: Theme.easeStandard
        }
    }
    Behavior on animUnitY {
        NumberAnimation {
            duration: Theme.hopDuration
            easing.type: Easing.Bezier
            easing.bezierCurve: Theme.easeStandard
        }
    }

    x: originX + animUnitX * unitScale - width / 2
    y: originY + animUnitY * unitScale - height / 2

    // The piece is exactly the socket: same radius, so an occupied hole reads
    // as that hole filled in the owner's colour rather than a token sitting on
    // top of one.
    //
    // Selection is carried by weight of colour alone — no ring. A selected
    // piece is the only fully saturated thing on the board.
    //
    // The washed-back state is a *blend against the board*, not `opacity`.
    // A genuinely translucent piece lets whatever sits behind it show through,
    // and two things do: the camp triangle, whose sharp edge cuts across the
    // boundary sockets and split those pieces diagonally, and the proposed move
    // path. Blending to an opaque colour makes a piece look the same wherever
    // it stands.
    readonly property color _shown: root.highlighted
        ? root.pieceColor
        : Qt.tint(Theme.boardBackground,
                  Qt.rgba(root.pieceColor.r, root.pieceColor.g,
                          root.pieceColor.b, Theme.pieceOpacity))

    Rectangle {
        anchors.centerIn: parent
        width: root.unitScale * Theme.socketRatio * 2
        height: width
        radius: width / 2
        color: root._shown
        // The rim is the piece's own colour, darkened — it defines the edge
        // without adding a second hue, and works for red, yellow and green
        // alike where one fixed grey would not.
        border.width: Math.max(1, root.unitScale * Theme.socketStroke)
        border.color: Qt.darker(root._shown, 1.3)
        antialiasing: true

        Behavior on color {
            ColorAnimation {
                duration: Theme.fadeDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeStandard
            }
        }
        Behavior on border.color {
            ColorAnimation {
                duration: Theme.fadeDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeStandard
            }
        }
    }
}
