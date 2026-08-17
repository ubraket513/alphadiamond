import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// The one dialog shell every popup in the app uses.
//
// The Basic style ships an unstyled Dialog whose title and buttons inherit
// whatever the platform palette happens to be — which is where the earlier
// unreadable pop-ups came from. Everything visible here is therefore painted
// explicitly: opaque surface, themed title, themed body text, and footer
// buttons built from ActionButton rather than `standardButtons`.
Dialog {
    id: root

    // Text shown under the title. Wraps, never elides, never clips.
    property string message: ""
    property string acceptText: "OK"
    property string rejectText: "Cancel"
    property bool showAccept: true
    property bool showReject: true
    property bool acceptEnabled: true
    property color accent: Theme.selection

    // Extra content below the message.
    default property alias body: extra.data

    anchors.centerIn: Overlay.overlay
    modal: true
    closePolicy: Popup.CloseOnEscape
    padding: 0

    implicitWidth: Math.max(400, layout.implicitWidth)

    // An opaque scrim: without it the dialog reads against the board lattice
    // and the message becomes hard to pick out.
    Overlay.modal: Rectangle {
        color: Qt.rgba(0.09, 0.09, 0.11, 0.55)
    }

    background: Rectangle {
        color: Theme.surface
        radius: Theme.radiusMedium
        border.width: 1
        border.color: Theme.borderStrong
    }

    // The Basic style's own header/footer are replaced wholesale.
    header: null
    footer: null

    contentItem: ColumnLayout {
        id: layout
        spacing: 0

        ColumnLayout {
            Layout.fillWidth: true
            Layout.margins: Theme.spacingLarge + 4
            spacing: Theme.spacing

            Text {
                Layout.fillWidth: true
                visible: root.title !== ""
                text: root.title
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontLarge
                font.weight: Theme.weightBold
                wrapMode: Text.WordWrap
            }

            Text {
                Layout.fillWidth: true
                visible: root.message !== ""
                text: root.message
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                lineHeight: 1.35
                wrapMode: Text.WordWrap
            }

            ColumnLayout {
                id: extra
                Layout.fillWidth: true
                Layout.topMargin: children.length > 0 ? Theme.spacing : 0
                spacing: Theme.spacing
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 1
            color: Theme.border
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.margins: Theme.spacingLarge
            spacing: Theme.spacing

            Item { Layout.fillWidth: true }

            ActionButton {
                visible: root.showReject
                text: root.rejectText
                implicitWidth: Math.max(96, implicitContentWidth + 28)
                onClicked: root.reject()
            }

            ActionButton {
                visible: root.showAccept
                text: root.acceptText
                primary: true
                accent: root.accent
                enabled: root.acceptEnabled
                implicitWidth: Math.max(96, implicitContentWidth + 28)
                onClicked: root.accept()
            }
        }
    }
}
