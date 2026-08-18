import QtQuick
import Style

// One playable hole, drawn as a hollow socket. Rendering + hit testing only;
// legality is decided in Python.
//
// A hole shows only what is actionable *now*: the socket, and whether the
// selected piece may land here. Rings marking the last move and the proposal's
// endpoints were removed -- they outlived their usefulness once the move path
// itself was drawn on the board, and three concentric rings around one hole
// read as noise rather than history.
//
// The socket and the piece share one radius (`Theme.socketRatio`), and the
// ring is stroked *inside* the socket bounds, so an occupied hole is the socket
// filled edge to edge in the owner's colour with no ring peeking out.
Item {
    id: root

    required property int positionId
    required property bool isSelected
    required property bool isLegalStep
    required property bool isLegalJump

    property real unitScale: 1
    property bool interactive: true

    signal clicked(int positionId)

    // Step and jump look identical: both are simply somewhere this piece can
    // go, and splitting them cost a second colour for no decision the operator
    // actually makes differently.
    readonly property bool isLegal: isLegalStep || isLegalJump
    readonly property real socketRadius: unitScale * Theme.socketRatio

    // The socket itself. `border` is drawn inside the item's bounds, so the
    // outer edge lands exactly on socketRadius.
    Rectangle {
        id: socket
        anchors.centerIn: parent
        width: root.socketRadius * 2
        height: width
        radius: width / 2
        // Opaque, not transparent: an empty hole is a hole in the board, so it
        // takes the board's own colour rather than letting the camp triangle
        // behind it show through. It also gives the move path something solid
        // to be drawn over.
        color: root.isLegal
               ? Qt.tint(Theme.boardBackground,
                         Qt.rgba(Theme.legalMove.r, Theme.legalMove.g,
                                 Theme.legalMove.b, Theme.ghostAlpha))
               : Theme.boardBackground
        border.width: Math.max(1, root.unitScale * Theme.socketStroke)
        border.color: root.isLegal ? Theme.legalMove
                                   : (hover.containsMouse && root.interactive
                                      ? Theme.borderStrong : Theme.hole)
        antialiasing: true

        Behavior on color {
            ColorAnimation {
                duration: Theme.fadeDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeEmphasized
            }
        }
        Behavior on border.color {
            ColorAnimation {
                duration: Theme.fadeDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeEmphasized
            }
        }
    }

    MouseArea {
        id: hover
        anchors.centerIn: parent
        width: Math.max(20, root.socketRadius * 2)
        height: width
        enabled: root.interactive
        cursorShape: root.interactive ? Qt.PointingHandCursor : Qt.ArrowCursor
        hoverEnabled: true
        onClicked: root.clicked(root.positionId)
    }
}
