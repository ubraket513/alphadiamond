import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    required property var controller
    title: "PLAYERS"

    Repeater {
        model: root.controller.playerModel

        delegate: RowLayout {
            id: row

            required property string name
            required property string kindLabel
            required property string color
            required property bool   isCurrent
            required property int    homeCount
            required property bool   hasFinished

            Layout.fillWidth: true
            spacing: Theme.spacing

            Rectangle {
                width: 4
                Layout.preferredHeight: 26
                radius: 2
                color: row.isCurrent ? Theme.selection : "transparent"
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
                font.bold: row.isCurrent
                elide: Text.ElideRight
            }

            Text {
                text: row.kindLabel
                color: Theme.textMuted
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
            }

            Text {
                text: row.homeCount + "/10"
                color: row.hasFinished ? Theme.legalStep : Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.bold: row.hasFinished
            }
        }
    }

}
