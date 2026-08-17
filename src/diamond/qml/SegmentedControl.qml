import QtQuick
import QtQuick.Layouts
import Style

// A row of mutually exclusive options in a single bordered track.
//
// The pattern is the one used for small closed sets in setup forms (see
// HelloFresh's "Number of People" and Cursor's team-size picker): all options
// visible at once with the active one filled, so the choice and its
// alternatives read in a single glance — better than a combo box for 2-3 items.
Item {
    id: root

    property var options: []          // [{ value, label }]
    property var currentValue: null
    signal picked(var value)

    implicitHeight: 36
    implicitWidth: track.implicitWidth

    Rectangle {
        id: track
        anchors.fill: parent
        radius: Theme.radiusSmall
        color: Theme.surfaceAlt
        border.width: 1
        border.color: Theme.border
        implicitWidth: row.implicitWidth + 8

        RowLayout {
            id: row
            anchors.fill: parent
            anchors.margins: 3
            spacing: 3

            Repeater {
                model: root.options

                Rectangle {
                    required property var modelData

                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumWidth: 64

                    readonly property bool active: modelData.value === root.currentValue

                    radius: Theme.radiusSmall - 1
                    color: active ? Theme.selection
                                  : (hover.hovered ? Theme.surface : "transparent")
                    border.width: active ? 0 : 1
                    border.color: hover.hovered ? Theme.border : "transparent"

                    Text {
                        anchors.centerIn: parent
                        text: parent.modelData.label
                        color: parent.active ? Theme.textOnDark : Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: parent.active ? Theme.weightMedium : Theme.weightRegular
                    }

                    HoverHandler { id: hover }
                    TapHandler { onTapped: root.picked(parent.modelData.value) }
                }
            }
        }
    }
}
