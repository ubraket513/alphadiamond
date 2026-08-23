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
                  ? "Estimated outlook · P" + root.controller.perspectivePlayerId
                  : "Unavailable for Min"
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTiny
        }
        SegmentedControl {
            visible: root.controller.analysisAvailable
            implicitWidth: 108
            implicitHeight: 26
            options: [{ value: 1, label: "P1" }, { value: 2, label: "P2" }]
            currentValue: root.controller.perspectivePlayerId
            onPicked: value => root.controller.setPerspectivePlayerId(value)
        }
    }

    TelemetryChart {
        points: root.controller.positionTelemetry
        firstKey: "nnEstimate"
        secondKey: "mctsEstimate"
        firstLabel: "NN estimate"
        secondLabel: "MCTS estimate"
        minimum: 0
        maximum: 1
        percent: true
        visible: root.controller.analysisAvailable
    }
}

