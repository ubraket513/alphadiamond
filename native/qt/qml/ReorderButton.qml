import QtQuick
import Style

// A small square button carrying a chevron.
//
// The glyph comes from QtAwesome's bundled Codicons rather than the text
// typeface, which has no arrow characters at all.
Item {
    id: root

    property bool up: true
    signal clicked()

    // `enabled` is inherited from Item; redeclaring it here shadowed the base
    // property and left the pointer handlers reading a different one.

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

    Icon {
        anchors.centerIn: parent
        name: root.up ? "msc.chevron-up" : "msc.chevron-down"
        size: 14
        color: Theme.text
        opacity: root.enabled ? 1.0 : 0.35
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
