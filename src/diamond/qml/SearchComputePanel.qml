import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    objectName: "searchComputePanel"
    required property var controller
    title: "Search Compute"

    readonly property var metrics: root.controller.latestSearchCompute
    readonly property bool hasMetrics: metrics && metrics.totalMs !== undefined

    function milliseconds(value) { return Number(value || 0).toFixed(1) + " ms" }
    function rate(value) { return String(Math.round(Number(value || 0))) + "/s" }

    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 8
        radius: 4
        color: Theme.border
        clip: true
        visible: root.hasMetrics

        Row {
            anchors.fill: parent
            Rectangle {
                width: parent.width * Number(root.metrics.neuralFraction || 0)
                height: parent.height
                color: Theme.accent
            }
            Rectangle {
                width: parent.width * Number(root.metrics.mctsRulesFraction || 0)
                height: parent.height
                color: Theme.systemOrange
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        visible: root.hasMetrics
        Text {
            Layout.fillWidth: true
            text: "Neural Network  " + (Number(root.metrics.neuralFraction || 0) * 100).toFixed(1) + "%"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTiny
        }
        Text {
            text: "MCTS / Rules  " + (Number(root.metrics.mctsRulesFraction || 0) * 100).toFixed(1) + "%"
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTiny
        }
    }

    GridLayout {
        Layout.fillWidth: true
        columns: 4
        columnSpacing: Theme.spacing
        rowSpacing: 3
        visible: root.hasMetrics

        Repeater {
            model: [
                { label: "Total", value: root.milliseconds(root.metrics.totalMs) },
                { label: "NN", value: root.milliseconds(root.metrics.neuralMs) },
                { label: "MCTS", value: root.milliseconds(root.metrics.mctsRulesMs) },
                { label: "Sims", value: String(root.metrics.simulations || 0) },
                { label: "Eval calls", value: String(root.metrics.evaluatorCalls || 0) },
                { label: "Nodes", value: String(root.metrics.nodes || 0) },
                { label: "Sims/sec", value: root.rate(root.metrics.simulationsPerSecond) },
                { label: "Avg NN", value: root.milliseconds(root.metrics.averageNeuralEvaluationMs) }
            ]
            delegate: ColumnLayout {
                required property var modelData
                Layout.fillWidth: true
                spacing: 0
                Text {
                    text: parent.modelData.label
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                }
                Text {
                    text: parent.modelData.value
                    color: Theme.text
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                    font.weight: Theme.weightMedium
                }
            }
        }
    }

    Text {
        Layout.fillWidth: true
        visible: !root.hasMetrics
        text: root.controller.analysisAvailable
              ? "Waiting for a completed search…"
              : "Compute analysis is unavailable for Min."
        color: Theme.textFaint
        font.family: Theme.fontFamily
        font.pixelSize: Theme.fontSmall
    }
}
