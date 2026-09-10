import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    objectName: "aiMovePanel"
    required property var controller
    title: "AI Move"
    Layout.minimumHeight: implicitHeight
    Layout.maximumHeight: implicitHeight

    ActionButton {
        objectName: "confirmAiMoveButton"
        Layout.fillWidth: true
        text: "Confirm"
        primary: true
        enabled: root.controller.hasProposal && root.controller.proposalIsAi
                 && root.controller.canConfirm
        onClicked: root.controller.confirmProposal()
    }
}
