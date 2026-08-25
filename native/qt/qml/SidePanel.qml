import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

// Left-hand operator console. Restrained so the board stays the centrepiece.
//
// The inner column is deliberately wrapped in a Flickable of fixed height: a
// bare ColumnLayout propagates its minimum height outwards, which would stretch
// the whole window row (and the board with it) whenever a panel grows. Here the
// column takes the panel height when the content fits and scrolls only when it
// genuinely cannot fit.
Item {
    id: root
    objectName: "analysisConsole"

    required property var controller

    implicitWidth: Theme.panelWidth

    Flickable {
        id: flick
        anchors.fill: parent
        clip: true
        contentWidth: column.width
        contentHeight: column.height
        boundsBehavior: Flickable.StopAtBounds
        flickableDirection: Flickable.VerticalFlick

        // While the panel collapses, keep the content's *right* edge pinned to
        // the panel's, so it slides out of view instead of being clipped in
        // place. The section titles are right-aligned, so they stay legible
        // for as long as there is any panel left.
        contentX: Math.max(0, column.width - width)

        ScrollBar.vertical: PanelScrollBar {}

        ColumnLayout {
            id: column
            // Pinned to the full panel width rather than the Flickable's, so
            // collapsing the panel clips the content instead of relaying it
            // out — otherwise every frame of the animation rewraps the text.
            width: Theme.panelWidth
            height: Math.max(implicitHeight, root.height)
            spacing: Theme.spacing

            GamePanel { controller: root.controller }
            SearchComputePanel { controller: root.controller }
            AiPanel { controller: root.controller }
        }
    }
}
