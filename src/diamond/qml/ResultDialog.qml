import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// Final standings.
//
// A ranked list with a medal chip per row rather than a podium graphic: this
// is an operator console, and the pattern that suits it is the compact
// leaderboard row (rank chip, identity, result) seen in Binance's ranking and
// Transit's contributor board — dense, scannable, no celebration furniture.
AppDialog {
    id: root

    signal rematchRequested()

    title: "Match over"
    message: controller.playerCount === 2
             ? "Final result."
             : "Every place is settled — the last seat placed without having to finish."
    acceptText: "New match"
    rejectText: "Close"
    implicitWidth: 460

    onAccepted: rematchRequested()

    function medalColor(place) {
        return place === 1 ? "#B8860B"      // gold, darkened to stay legible
             : place === 2 ? "#78808A"      // silver
             : place === 3 ? "#8C5A33"      // bronze
             : Theme.textFaint
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 0

        Repeater {
            model: controller.standings

            Item {
                required property var modelData

                Layout.fillWidth: true
                implicitHeight: 52

                Rectangle {
                    anchors.fill: parent
                    radius: Theme.radiusSmall
                    color: modelData.place === 1 ? Theme.surfaceAlt : "transparent"
                }

                Rectangle {
                    visible: modelData.place > 1
                    width: parent.width
                    anchors.top: parent.top
                    height: 1
                    color: Theme.border
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: Theme.spacing
                    anchors.rightMargin: Theme.spacing
                    spacing: Theme.spacing

                    // Rank chip
                    Rectangle {
                        implicitWidth: 34
                        implicitHeight: 22
                        radius: Theme.radiusSmall
                        color: root.medalColor(modelData.place)

                        Text {
                            anchors.centerIn: parent
                            text: modelData.placeLabel
                            color: Theme.textOnDark
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontSmall
                            font.weight: Theme.weightBold
                        }
                    }

                    Rectangle {
                        width: 16
                        height: 16
                        radius: 8
                        color: modelData.color
                        border.width: 1
                        border.color: Qt.darker(modelData.color, 1.4)
                    }

                    Text {
                        Layout.fillWidth: true
                        text: modelData.name
                        color: Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: modelData.place === 1 ? Theme.weightBold
                                                           : Theme.weightRegular
                        elide: Text.ElideRight
                    }

                    Text {
                        text: modelData.isAi ? "Agent" : "Human"
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontSmall
                    }
                }
            }
        }
    }
}
