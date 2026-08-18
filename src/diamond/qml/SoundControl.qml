import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Shapes
import Style

// Move-sound control for the title bar: a speaker button that opens a small
// volume popover.
//
// Pattern follows the toolbar audio controls in Canva and Adobe Express — an
// icon button, and on click a compact popover holding [mute] [slider]
// [readout]. Keeping the slider out of the bar itself matters here: the title
// bar is chrome, and a permanently visible slider would compete with the menus
// for a setting that gets adjusted once and then left alone.
Item {
    id: root

    required property var controller

    readonly property bool on: controller.soundEnabled
    readonly property bool broken: !controller.soundAvailable
    readonly property real level: controller.soundVolume

    implicitWidth: 38
    implicitHeight: Theme.titleBarHeight

    Rectangle {
        anchors.centerIn: parent
        width: 30
        height: 26
        radius: Theme.radiusSmall
        color: popup.opened ? Theme.systemGray5
                            : (hover.hovered ? Theme.systemGray6 : "transparent")
    }

    // Speaker glyph. The number of arcs tracks the level, so the button alone
    // reports roughly how loud it is without opening the popover.
    Shape {
        anchors.centerIn: parent
        width: 18
        height: 16
        opacity: root.broken ? 0.4 : 1.0

        ShapePath {
            fillColor: (root.on && !root.broken) ? Theme.text : Theme.textFaint
            strokeColor: "transparent"
            startX: 1; startY: 6
            PathLine { x: 5;  y: 6 }
            PathLine { x: 9;  y: 2 }
            PathLine { x: 9;  y: 14 }
            PathLine { x: 5;  y: 10 }
            PathLine { x: 1;  y: 10 }
        }

        ShapePath {
            strokeColor: Theme.accent
            strokeWidth: 1.5
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            startX: (root.on && !root.broken && root.level > 0.02) ? 11.5 : 0
            startY: (root.on && !root.broken && root.level > 0.02) ? 5 : 0
            PathArc {
                x: 11.5; y: 11
                radiusX: 3.5; radiusY: 3.5
                direction: PathArc.Clockwise
            }
        }

        ShapePath {
            strokeColor: Theme.accent
            strokeWidth: 1.5
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            startX: (root.on && !root.broken && root.level > 0.55) ? 14 : 0
            startY: (root.on && !root.broken && root.level > 0.55) ? 3 : 0
            PathArc {
                x: 14; y: 13
                radiusX: 6; radiusY: 6
                direction: PathArc.Clockwise
            }
        }

        // Muted or unavailable: a slash across the speaker.
        ShapePath {
            strokeColor: root.broken ? Theme.textFaint : Theme.danger
            strokeWidth: 1.6
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            startX: (root.on && !root.broken) ? 0 : 2
            startY: (root.on && !root.broken) ? 0 : 15
            PathLine {
                x: (root.on && !root.broken) ? 0 : 17
                y: (root.on && !root.broken) ? 0 : 1
            }
        }
    }

    HoverHandler { id: hover; cursorShape: Qt.PointingHandCursor }
    TapHandler { onTapped: popup.opened ? popup.close() : popup.open() }

    ToolTip.visible: hover.hovered && !popup.opened
    ToolTip.text: root.broken
                  ? (controller.soundStatus || "Move sound unavailable.")
                  : "Move sound — " + Math.round(root.level * 100) + "%"

    Popup {
        id: popup

        y: root.height + 4
        x: -width + root.width + 8
        padding: 0
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent

        background: Rectangle {
            color: Theme.surface
            radius: Theme.radiusMedium
            border.width: 1
            border.color: Theme.borderStrong
        }

        contentItem: ColumnLayout {
            spacing: Theme.spacing

            RowLayout {
                Layout.margins: Theme.spacing + 2
                Layout.bottomMargin: 0
                spacing: Theme.spacing

                Text {
                    text: "Move sound"
                    color: Theme.text
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                    font.weight: Theme.weightMedium
                }
                Item { Layout.fillWidth: true; Layout.minimumWidth: 24 }
                Text {
                    text: root.broken ? "unavailable" : (root.on ? "on" : "muted")
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                }
            }

            RowLayout {
                Layout.margins: Theme.spacing + 2
                Layout.topMargin: 0
                spacing: Theme.spacing
                enabled: !root.broken

                // Mute button, left of the slider (Canva / Adobe Express).
                Rectangle {
                    implicitWidth: 30
                    implicitHeight: 26
                    radius: Theme.radiusSmall
                    color: muteHover.hovered ? Theme.systemGray6 : "transparent"
                    border.width: 1
                    border.color: Theme.border

                    Text {
                        anchors.centerIn: parent
                        text: root.on ? "on" : "off"
                        color: root.on ? Theme.accent : Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontTiny
                        font.weight: Theme.weightMedium
                    }

                    HoverHandler { id: muteHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: root.controller.setSoundEnabled(!root.on) }
                }

                Slider {
                    id: slider
                    Layout.fillWidth: true
                    Layout.minimumWidth: 132
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
                        width: 16
                        height: 16
                        radius: 8
                        color: Theme.surface
                        border.width: 1
                        border.color: slider.pressed ? Theme.accent : Theme.borderStrong
                    }
                }

                // Numeric readout, as in the Canva audio popover.
                Text {
                    Layout.preferredWidth: 34
                    horizontalAlignment: Text.AlignRight
                    text: Math.round(root.level * 100) + "%"
                    color: Theme.textMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                }
            }
        }
    }
}
