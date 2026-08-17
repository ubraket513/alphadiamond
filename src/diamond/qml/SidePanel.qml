import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// Right-hand operator console. Restrained so the board stays the centrepiece.
//
// The inner column is deliberately wrapped in a Flickable of fixed height: a
// bare ColumnLayout propagates its minimum height outwards, which would stretch
// the whole window row (and the board with it) whenever a panel grows. Here the
// column takes the panel height when the content fits — so the history list
// absorbs the slack — and scrolls only when it genuinely cannot fit.
Item {
    id: root

    required property var controller

    implicitWidth: Theme.panelWidth

    Flickable {
        id: flick
        anchors.fill: parent
        clip: true
        contentWidth: width
        contentHeight: column.height
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

        ColumnLayout {
            id: column
            width: flick.width
            height: Math.max(implicitHeight, root.height)
            spacing: Theme.spacing

            GamePanel   { controller: root.controller }
            PlayerPanel { controller: root.controller }
            MovePanel   { controller: root.controller }
            AiPanel     { controller: root.controller }

            HistoryPanel {
                controller: root.controller
                Layout.fillHeight: true
                Layout.minimumHeight: 96
            }
        }
    }
}
