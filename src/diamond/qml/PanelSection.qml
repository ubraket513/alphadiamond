import QtQuick
import QtQuick.Layouts
import Style

// A titled block in the side panel. Plain surface, one hairline border, no
// gradients or shadows.
Rectangle {
    id: root

    property string title: ""
    default property alias content: body.data

    Layout.fillWidth: true
    implicitHeight: layout.implicitHeight + 2 * Theme.spacingLarge
    color: Theme.surface
    border.width: 1
    border.color: Theme.border
    radius: Theme.radiusMedium

    ColumnLayout {
        id: layout
        anchors.fill: parent
        anchors.margins: Theme.spacingLarge
        spacing: Theme.spacing

        // The section title is right-aligned while its content stays left, so
        // the titles form a quiet right-hand rail down the panel instead of
        // competing with the content for the same starting edge.
        Text {
            Layout.fillWidth: true
            visible: root.title !== ""
            text: root.title
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            font.letterSpacing: 1.2
            font.weight: Theme.weightBold
            horizontalAlignment: Text.AlignRight
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: Theme.spacing
        }
    }
}
