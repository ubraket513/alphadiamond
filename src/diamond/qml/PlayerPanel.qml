import QtQuick
import QtQuick.Layouts
import Style

// Live scoreboard: seats in turn order, progress toward home, and the place
// each one took once they got there.
PanelSection {
    id: root
    required property var controller
    title: "PLAYERS"

    function medalColor(place) {
        return place === 1 ? "#B8860B"
             : place === 2 ? "#78808A"
             : place === 3 ? "#8C5A33"
             : Theme.textFaint
    }

    Repeater {
        model: root.controller.playerModel

        delegate: RowLayout {
            id: row

            required property string name
            required property string kindLabel
            required property string color
            required property bool   isCurrent
            required property int    homeCount
            required property int    campSize
            required property bool   hasFinished
            required property int    turnIndex
            required property int    place
            required property string placeLabel

            Layout.fillWidth: true
            spacing: Theme.spacing
            // A seat that has placed is out of the rotation; dim it so the
            // active seats stay the ones that read first.
            opacity: row.place > 0 ? 0.65 : 1.0

            Rectangle {
                width: 4
                Layout.preferredHeight: 26
                radius: 2
                color: row.isCurrent ? Theme.selection : "transparent"
            }

            Text {
                text: row.turnIndex
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontTiny
                Layout.preferredWidth: 8
            }

            Rectangle {
                width: 16; height: 16; radius: 8
                color: row.color
                border.width: 1
                border.color: Theme.lattice
            }

            Text {
                Layout.fillWidth: true
                text: row.name
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                font.weight: row.isCurrent ? Theme.weightBold : Theme.weightRegular
                elide: Text.ElideRight
            }

            Text {
                visible: row.place === 0
                text: row.kindLabel
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
            }

            // Progress while playing, final place once home.
            Text {
                visible: row.place === 0
                text: row.homeCount + "/" + row.campSize
                color: row.hasFinished ? Theme.legalStep : Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.weight: row.hasFinished ? Theme.weightBold : Theme.weightRegular
            }

            Rectangle {
                visible: row.place > 0
                implicitWidth: 34
                implicitHeight: 20
                radius: Theme.radiusSmall
                color: root.medalColor(row.place)

                Text {
                    anchors.centerIn: parent
                    text: row.placeLabel
                    color: Theme.textOnDark
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                    font.weight: Theme.weightBold
                }
            }
        }
    }
}
