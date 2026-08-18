import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// Move-sound settings, opened from View ▸ Sounds.
//
// Layout is the toolbar audio control from Canva and Adobe Express — a mute
// button, a slider and a numeric readout on one row — given a dialog rather
// than a popover now that it is reached from a menu instead of a bar button.
AppDialog {
    id: root

    required property var controller

    readonly property bool on: root.controller.soundEnabled
    readonly property bool broken: !root.controller.soundAvailable
    readonly property real level: root.controller.soundVolume

    title: "Sounds"
    message: root.broken
             ? (root.controller.soundStatus || "The move sound is unavailable on this machine.")
             : "A sound plays for each hop of a confirmed move — once for a step, "
               + "once per landing for a chain of jumps.

"
               + "Arrow keys adjust the volume, Space mutes."
    acceptText: "Done"
    showReject: false
    implicitWidth: 440

    // Step used by the arrow keys. Twenty presses cross the full range, which
    // is fine enough to tune by ear without becoming tedious.
    readonly property real volumeStep: 0.05

    function nudgeVolume(delta) {
        if (root.broken)
            return
        root.controller.setSoundVolume(
            Math.max(0, Math.min(1, root.level + delta)))
        root.controller.previewSound()
    }

    // Arrow keys adjust, Space mutes. Shortcuts rather than `Keys` handlers for
    // the same reason as the Enter binding in AppDialog: a Popup is not in the
    // focus chain, so it never sees the key itself. Scoping them to
    // `root.visible` keeps them from firing while the dialog is closed.
    Shortcut {
        sequences: ["Right", "Up"]
        enabled: root.visible && !root.broken
        onActivated: root.nudgeVolume(root.volumeStep)
    }
    Shortcut {
        sequences: ["Left", "Down"]
        enabled: root.visible && !root.broken
        onActivated: root.nudgeVolume(-root.volumeStep)
    }
    Shortcut {
        sequence: "Space"
        enabled: root.visible && !root.broken
        onActivated: root.controller.setSoundEnabled(!root.on)
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: Theme.spacing
        enabled: !root.broken

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacing

            Text {
                text: "Move sound"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
            }
            Item { Layout.fillWidth: true }
            Text {
                text: root.broken ? "unavailable" : (root.on ? "On" : "Muted")
                color: root.on && !root.broken ? Theme.accent : Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.weight: Theme.weightMedium
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacing

            ActionButton {
                text: root.on ? "Mute" : "Unmute"
                implicitWidth: 88
                onClicked: root.controller.setSoundEnabled(!root.on)
            }

            Slider {
                id: slider
                Layout.fillWidth: true
                from: 0
                to: 1
                value: root.level
                onMoved: root.controller.setSoundVolume(value)
                // Preview on release only: previewing on every pixel of the
                // drag would stutter the clip against itself.
                onPressedChanged: if (!pressed) root.controller.previewSound()

                background: Rectangle {
                    x: slider.leftPadding
                    y: slider.topPadding + slider.availableHeight / 2 - height / 2
                    width: slider.availableWidth
                    height: 4
                    radius: 2
                    color: Theme.systemGray5

                    Rectangle {
                        width: slider.visualPosition * parent.width
                        height: parent.height
                        radius: 2
                        color: root.on ? Theme.accent : Theme.systemGray3
                    }
                }

                handle: Rectangle {
                    x: slider.leftPadding + slider.visualPosition
                       * (slider.availableWidth - width)
                    y: slider.topPadding + slider.availableHeight / 2 - height / 2
                    width: 18
                    height: 18
                    radius: 9
                    color: Theme.surface
                    border.width: 1
                    border.color: slider.pressed ? Theme.accent : Theme.borderStrong
                }
            }

            Text {
                Layout.preferredWidth: 40
                horizontalAlignment: Text.AlignRight
                text: Math.round(root.level * 100) + "%"
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
            }
        }
    }
}
