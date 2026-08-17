import QtQuick
import QtQuick.Controls.Basic
import Style

// Flat, calm button. `primary` marks the one action the current phase expects.
Button {
    id: root

    property bool primary: false
    property color accent: Theme.selection

    implicitHeight: 34
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fontBody

    background: Rectangle {
        radius: Theme.radiusSmall
        color: !root.enabled
               ? Theme.surfaceAlt
               : root.primary
                 ? (root.down ? Qt.darker(root.accent, 1.25) : root.accent)
                 : (root.down ? Theme.surfaceAlt : Theme.surface)
        border.width: 1
        border.color: root.enabled ? (root.primary ? root.accent : Theme.borderStrong)
                                   : Theme.border
    }

    contentItem: Text {
        text: root.text
        font: root.font
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
        color: !root.enabled ? Theme.textFaint
                             : root.primary ? Theme.textOnDark : Theme.text
    }
}
