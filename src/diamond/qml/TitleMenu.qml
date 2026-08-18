import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// One flat menu in the custom title bar (File, Edit, View, …).
//
// Built from a plain model rather than QtQuick.Controls `Menu` so the popup
// takes the app's own surface, border and typography — the Basic style's menu
// inherits the platform palette, which is what made the earlier dialogs
// unreadable.
//
// `model` is a list of entries:
//   { label, shortcut?, action, enabled? }   or   { separator: true }
Item {
    id: root

    property string text: ""
    property var model: []

    signal triggered(string action)

    implicitWidth: label.implicitWidth + 2 * Theme.spacing + 4
    implicitHeight: Theme.titleBarHeight

    Rectangle {
        anchors.centerIn: parent
        width: parent.width - 4
        height: 26
        radius: Theme.radiusSmall
        color: popup.opened ? Theme.systemGray5
                            : (hover.hovered ? Theme.systemGray6 : "transparent")
    }

    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: Theme.text
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontBody
    }

    HoverHandler { id: hover; cursorShape: Qt.PointingHandCursor }
    TapHandler { onTapped: popup.opened ? popup.close() : popup.open() }

    Popup {
        id: popup

        y: root.height
        x: 2
        padding: 4
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
            spacing: 0

            Repeater {
                model: root.model

                delegate: Item {
                    id: row
                    required property var modelData

                    readonly property bool isSeparator: modelData.separator === true
                    readonly property bool isEnabled:
                        !isSeparator && modelData.enabled !== false

                    Layout.fillWidth: true
                    Layout.preferredHeight: isSeparator ? 9 : 30
                    Layout.minimumWidth: 210

                    Rectangle {
                        visible: row.isSeparator
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - 2 * Theme.spacing
                        x: Theme.spacing
                        height: 1
                        color: Theme.border
                    }

                    Rectangle {
                        visible: !row.isSeparator && itemHover.hovered && row.isEnabled
                        anchors.fill: parent
                        anchors.margins: 1
                        radius: Theme.radiusSmall
                        color: Theme.systemGray6
                    }

                    RowLayout {
                        visible: !row.isSeparator
                        anchors.fill: parent
                        anchors.leftMargin: Theme.spacing
                        anchors.rightMargin: Theme.spacing
                        spacing: Theme.spacingLarge

                        Text {
                            text: row.modelData.label || ""
                            color: row.isEnabled ? Theme.text : Theme.textFaint
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontBody
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: row.modelData.shortcut || ""
                            color: Theme.textFaint
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontSmall
                        }
                    }

                    HoverHandler {
                        id: itemHover
                        enabled: row.isEnabled
                        cursorShape: Qt.PointingHandCursor
                    }
                    TapHandler {
                        enabled: row.isEnabled
                        onTapped: {
                            popup.close()
                            root.triggered(row.modelData.action || "")
                        }
                    }
                }
            }
        }
    }
}
