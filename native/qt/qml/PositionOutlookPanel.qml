import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    objectName: "positionOutlookPanel"
    required property var controller
    title: "Position Outlook"

    RowLayout {
        Layout.fillWidth: true
        Text {
            Layout.fillWidth: true
            text: root.controller.analysisAvailable
                  ? (root.controller.playerCount === 3
                     ? "Placement utility · moving player (−1 to +1)"
                     : "Estimated outlook · P" + root.controller.perspectivePlayerId)
                  : "Analysis unavailable"
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTiny
        }
        SegmentedControl {
            visible: root.controller.analysisAvailable && root.controller.playerCount === 2
            implicitWidth: 108
            implicitHeight: 26
            options: [{ value: 1, label: "P1" }, { value: 2, label: "P2" }]
            currentValue: root.controller.perspectivePlayerId
            onPicked: value => root.controller.setPerspectivePlayerId(value)
        }
    }

    TelemetryChart {
        objectName: "positionOutlookChart"
        accessibleName: "Position outlook"
        points: root.controller.positionTelemetry
        firstKey: root.controller.playerCount === 3 ? "nnValue" : "nnEstimate"
        secondKey: root.controller.playerCount === 3 ? "mctsValue" : "mctsEstimate"
        firstLabel: "NN estimate"
        secondLabel: "MCTS estimate"
        minimum: root.controller.playerCount === 3 ? -1 : 0
        maximum: 1
        percent: root.controller.playerCount === 2
        visible: root.controller.analysisAvailable
    }
}
