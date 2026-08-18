import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// Match setup: how many seats, who plays in what order, and which seat the
// agent drives.
//
// Layout follows the compact setup-form pattern (Cursor's "Set Up Your Team",
// Whereby's room-size step): a segmented control for the small closed choice,
// then a reorderable list for the sequence, all in one modal with a single
// confirming action. Reordering uses explicit up/down controls rather than
// drag-and-drop — with at most three rows the target is always one click away,
// and it stays keyboard- and screen-reader-reachable.
AppDialog {
    id: root

    property int playerCount: 3
    property var order: [1, 2, 3]     // seat ids, in turn order
    property int aiSeat: 3            // 0 = all human

    signal confirmed(var order, var aiSeats)

    title: "New match"
    message: "Set the seats and turn order. Starting a match ends the one in progress."
    acceptText: "Start match"
    implicitWidth: 520

    onAboutToShow: reset()

    // Re-seed from the live controller setup each time the dialog opens, so it
    // always reflects the match actually running.
    function reset() {
        playerCount = controller.playerCount
        order = controller.turnOrder.slice()
        var seats = controller.aiSeats
        aiSeat = seats.length > 0 ? seats[0] : 0
    }

    function defaultAiSeat(seats) {
        // Keep the agent on a seat that exists in this layout.
        return seats.indexOf(aiSeat) >= 0 ? aiSeat : seats[seats.length - 1]
    }

    function setCount(count) {
        if (count === playerCount)
            return
        playerCount = count
        var seats = []
        for (var i = 1; i <= count; i++)
            seats.push(i)
        order = seats
        aiSeat = defaultAiSeat(seats)
    }

    function move(index, delta) {
        var target = index + delta
        if (target < 0 || target >= order.length)
            return
        var next = order.slice()
        var carried = next[index]
        next[index] = next[target]
        next[target] = carried
        order = next
    }

    // Colours come from the engine, which owns SEAT_LAYOUTS: seat 2 is yellow
    // in a 3-player match but green head-to-head, so they depend on the count.
    readonly property var seatColors: controller.seatColorsFor(playerCount)

    function seatColor(seat) {
        return seatColors[seat - 1] || Theme.textFaint
    }

    onAccepted: confirmed(order, aiSeat > 0 ? [aiSeat] : [])

    ColumnLayout {
        Layout.fillWidth: true
        spacing: Theme.spacingLarge

        // -- seat count ---------------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                text: "Players"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.weight: Theme.weightMedium
            }

            SegmentedControl {
                Layout.fillWidth: true
                options: [
                    { value: 2, label: "2 players" },
                    { value: 3, label: "3 players" }
                ]
                currentValue: root.playerCount
                onPicked: (value) => root.setCount(value)
            }

            Text {
                Layout.fillWidth: true
                text: root.playerCount === 2
                      ? "Head to head: both camps sit opposite each other."
                      : "Three camps, 120° apart, on alternating sides of the hexagon."
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                wrapMode: Text.WordWrap
            }
        }

        // -- turn order ---------------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                text: "Turn order"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.weight: Theme.weightMedium
            }

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: seatColumn.implicitHeight + 2
                color: Theme.surfaceAlt
                radius: Theme.radiusSmall
                border.width: 1
                border.color: Theme.border

                ColumnLayout {
                    id: seatColumn
                    anchors.fill: parent
                    anchors.margins: 1
                    spacing: 0

                    Repeater {
                        model: root.order

                        Item {
                            required property int index
                            required property var modelData

                            Layout.fillWidth: true
                            implicitHeight: 46

                            Rectangle {
                                anchors.fill: parent
                                color: seatHover.hovered ? Theme.surface : "transparent"
                            }
                            HoverHandler { id: seatHover }

                            Rectangle {
                                visible: index > 0
                                width: parent.width - 2 * Theme.spacing
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                height: 1
                                color: Theme.border
                            }

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: Theme.spacing
                                anchors.rightMargin: 6
                                spacing: Theme.spacing

                                Text {
                                    text: (index + 1) + "."
                                    color: Theme.textFaint
                                    font.family: Theme.fontFamily
                                    font.pixelSize: Theme.fontSmall
                                    Layout.preferredWidth: 16
                                }

                                Rectangle {
                                    width: 14
                                    height: 14
                                    radius: 7
                                    color: root.seatColor(modelData)
                                    border.width: 1
                                    border.color: Qt.darker(color, 1.4)
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: "Player " + modelData
                                    color: Theme.text
                                    font.family: Theme.fontFamily
                                    font.pixelSize: Theme.fontBody
                                    font.weight: index === 0 ? Theme.weightMedium
                                                             : Theme.weightRegular
                                }

                                Text {
                                    visible: index === 0
                                    text: "goes first"
                                    color: Theme.textFaint
                                    font.family: Theme.fontFamily
                                    font.pixelSize: Theme.fontSmall
                                }

                                ReorderButton {
                                    up: true
                                    enabled: index > 0
                                    onClicked: root.move(index, -1)
                                }

                                ReorderButton {
                                    up: false
                                    enabled: index < root.order.length - 1
                                    onClicked: root.move(index, 1)
                                }
                            }
                        }
                    }
                }
            }
        }

        // -- agent seat ---------------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                text: "Agent plays"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                font.weight: Theme.weightMedium
            }

            SegmentedControl {
                Layout.fillWidth: true
                options: {
                    var opts = [{ value: 0, label: "Nobody" }]
                    for (var i = 1; i <= root.playerCount; i++)
                        opts.push({ value: i, label: "Player " + i })
                    return opts
                }
                currentValue: root.aiSeat
                onPicked: (value) => root.aiSeat = value
            }

            Text {
                Layout.fillWidth: true
                text: root.aiSeat === 0
                      ? "Every seat is recorded by the controller."
                      : "The controller confirms the agent's move, then plays it on the real board."
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                wrapMode: Text.WordWrap
            }
        }
    }
}
