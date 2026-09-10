import QtQuick
import QtQuick.Layouts
import Style

// A titled block in the side panel. Plain surface, one hairline border, no
// gradients or shadows.
Rectangle {
    id: root

    property string title: ""
    property string subtitle: ""
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

        RowLayout {
            Layout.fillWidth: true
            visible: root.title !== ""
            spacing: Theme.spacing

            Text {
                text: root.title
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.letterSpacing: 1.2
                font.weight: Theme.weightBold
            }
            Text {
                Layout.fillWidth: true
                text: root.subtitle
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontTiny
                horizontalAlignment: Text.AlignRight
                elide: Text.ElideRight
            }
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: Theme.spacing
        }
    }
}
