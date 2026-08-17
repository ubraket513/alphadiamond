import QtQuick
import QtQuick.Layouts
import Style

// Human move proposal: selection -> proposal -> confirmation -> commit.
// Nothing here changes game state; Confirm calls the controller.
PanelSection {
    id: root
    required property var controller
    title: "MOVE"

    readonly property bool proposedByHuman:
        root.controller.hasProposal && !root.controller.proposalIsAi

    Text {
        Layout.fillWidth: true
        visible: !root.proposedByHuman
        text: root.controller.isGameOver
              ? "The match is over."
              : root.controller.isCurrentPlayerAi
                ? "Player 3 is an agent — see the AI panel."
                : root.controller.selectedPosition >= 0
                  ? "Piece " + root.controller.selectedPosition
                    + " selected. Click a highlighted destination."
                  : "Click one of the current player's pieces to begin."
        color: Theme.textMuted
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontBody
        wrapMode: Text.WordWrap
    }

    ColumnLayout {
        Layout.fillWidth: true
        visible: root.proposedByHuman
        spacing: 6

        Text {
            text: "Proposed move"
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
                  + "   (" + root.controller.proposalHopCount + " hops)"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            wrapMode: Text.WordWrap
        }
    }

    // Hidden rather than disabled on an agent turn: these controls belong to
    // the human proposal flow and would only be noise there.
    RowLayout {
        Layout.fillWidth: true
        spacing: Theme.spacing
        visible: !root.controller.isCurrentPlayerAi && !root.controller.isGameOver

        ActionButton {
            Layout.fillWidth: true
            text: "Confirm  (Enter)"
            primary: true
            enabled: root.controller.canConfirm && !root.controller.proposalIsAi
            onClicked: root.controller.confirmProposal()
        }
        ActionButton {
            Layout.fillWidth: true
            text: "Cancel  (Esc)"
            enabled: root.controller.canCancel
            onClicked: root.controller.cancelProposal()
        }
    }
}
