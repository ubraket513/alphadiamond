import QtQuick
import QtQuick.Shapes
import Style

// A small square button carrying a drawn chevron.
//
// The glyph is drawn rather than typed: Google Sans Flex has no arrow
// characters, and a font-dependent icon is one missing codepoint away from
// rendering as a tofu box.
Item {
    id: root

    property bool up: true
    property bool enabled: true
    signal clicked()

    implicitWidth: 30
    implicitHeight: 28

    Rectangle {
        anchors.fill: parent
        radius: Theme.radiusSmall
        color: !root.enabled ? "transparent"
                             : (tap.pressed ? Theme.surfaceAlt : Theme.surface)
        border.width: 1
        border.color: root.enabled ? Theme.borderStrong : Theme.border
        opacity: root.enabled ? 1.0 : 0.45
    }

    Shape {
        anchors.centerIn: parent
        width: 10
        height: 6
        opacity: root.enabled ? 1.0 : 0.35

        ShapePath {
            strokeColor: Theme.text
            strokeWidth: 1.6
            capStyle: ShapePath.RoundCap
            joinStyle: ShapePath.RoundJoin
            fillColor: "transparent"

            // Chevron: down-pointing is the up-pointing one mirrored.
            startX: 0
            startY: root.up ? 6 : 0
            PathLine { x: 5; y: root.up ? 0 : 6 }
            PathLine { x: 10; y: root.up ? 6 : 0 }
        }
    }

    TapHandler {
        id: tap
        enabled: root.enabled
        onTapped: root.clicked()
    }

    HoverHandler {
        enabled: root.enabled
        cursorShape: Qt.PointingHandCursor
    }
}
