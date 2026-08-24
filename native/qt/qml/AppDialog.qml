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
    focus: true
    closePolicy: Popup.CloseOnEscape
    padding: 0

    // Enter confirms, matching Esc cancelling: a dialog opened from a keyboard
    // shortcut should be dismissable from the keyboard too.
    //
    // A Shortcut rather than `Keys.onReturnPressed`, because a Popup does not
    // sit in the focus chain the way an Item does and never sees the key. The
    // board's own Return shortcut stands down while a dialog is open (see
    // `dialogOpen` in Main.qml), so the two cannot both fire.
    Shortcut {
        sequences: ["Return", "Enter"]
        enabled: root.visible && root.showAccept && root.acceptEnabled
        onActivated: root.accept()
    }

    // Esc dismisses. `closePolicy` covers this when the popup holds focus, but
    // a focused control inside it can swallow the key first, so the shortcut
    // makes it deterministic.
    Shortcut {
        sequences: ["Escape"]
        enabled: root.visible
        onActivated: root.reject()
    }

    // Scales up from just under full size while fading in: a plain fade reads
    // as a slideshow, and a large scale reads as a cartoon. The exit is
    // quicker than the entrance, because a dismissed dialog should get out of
    // the way rather than be admired on the way out.
    enter: Transition {
        ParallelAnimation {
            NumberAnimation {
                property: "opacity"
                from: 0.0; to: 1.0
                duration: Theme.durationBase
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeEmphasized
            }
            NumberAnimation {
                property: "scale"
                from: 0.96; to: 1.0
                duration: Theme.durationBase
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeEmphasized
            }
        }
    }

    exit: Transition {
        ParallelAnimation {
            NumberAnimation {
                property: "opacity"
                from: 1.0; to: 0.0
                duration: Theme.durationFast
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeStandard
            }
            NumberAnimation {
                property: "scale"
                from: 1.0; to: 0.98
                duration: Theme.durationFast
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeStandard
            }
        }
    }

    implicitWidth: Math.max(400, layout.implicitWidth)

    // An opaque scrim: without it the dialog reads against the board lattice
    // and the message becomes hard to pick out.
    Overlay.modal: Rectangle {
        color: Qt.rgba(0.09, 0.09, 0.11, 0.55)

        // The scrim fades with the dialog rather than snapping in behind it.
        Behavior on opacity {
            NumberAnimation { duration: Theme.durationBase }
        }
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
