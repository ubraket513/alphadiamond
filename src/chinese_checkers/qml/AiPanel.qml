import QtQuick
import QtQuick.Layouts
import Style

// AI status and proposal. Only metadata the agent actually reported is shown —
// no placeholder evaluation numbers.
PanelSection {
    id: root
    required property var controller
    title: "AI AGENT"

    readonly property bool proposedByAi:
        root.controller.hasProposal && root.controller.proposalIsAi

    GridLayout {
        Layout.fillWidth: true
        columns: 2
        columnSpacing: Theme.spacingLarge
        rowSpacing: 4

        Text {
            text: "Status"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        RowLayout {
            spacing: 6
            Rectangle {
                width: 8; height: 8; radius: 4
                color: root.controller.aiStatus === "Thinking…" ? Theme.thinking
                     : root.proposedByAi ? Theme.proposal
                     : Theme.textFaint
            }
            Text {
                text: root.controller.aiStatus
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
            }
        }

        Text {
            text: "Agent"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        Text {
            text: root.controller.aiAgentName
            color: Theme.text
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontBody
        }
    }

    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 1
        color: Theme.border
        visible: root.proposedByAi
    }

    ColumnLayout {
        Layout.fillWidth: true
        visible: root.proposedByAi
        spacing: 6

        Text {
            text: "Suggested move"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        Text {
            text: root.controller.proposalSummary
            color: Theme.text
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTitle
        }
        Text {
            Layout.fillWidth: true
            visible: root.controller.proposalIsMultiHop
            text: "Path:  " + root.controller.proposalPath
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            wrapMode: Text.WordWrap
        }

        Repeater {
            model: root.controller.aiDetails
            delegate: RowLayout {
                required property var modelData
                Layout.fillWidth: true
                Text {
                    Layout.fillWidth: true
                    text: modelData.label
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                }
                Text {
                    text: modelData.value
                    color: Theme.textMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                }
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacing

        ActionButton {
            Layout.fillWidth: true
            text: "Confirm AI Move"
            primary: true
            enabled: root.proposedByAi
            onClicked: root.controller.confirmProposal()
        }
        ActionButton {
            Layout.fillWidth: true
            text: "Think Again"
            enabled: root.proposedByAi
            onClicked: root.controller.thinkAgain()
        }
    }
}
