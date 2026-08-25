import QtQuick
import Style

PanelSection {
    id: root
    objectName: "decisionValuePanel"
    required property var controller
    title: "Decision Value"

    TelemetryChart {
        points: root.controller.decisionTelemetry
        firstKey: "nnValue"
        secondKey: "mctsQ"
        firstLabel: "NN value"
        secondLabel: "Selected MCTS Q"
        minimum: -1
        maximum: 1
        visible: root.controller.analysisAvailable
    }

    Text {
        visible: !root.controller.analysisAvailable
        text: "Native Min analysis is not available."
        color: Theme.textFaint
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontSmall
    }
}

