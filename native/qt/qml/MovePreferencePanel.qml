import QtQuick
import Style

PanelSection {
    id: root
    objectName: "movePreferencePanel"
    required property var controller
    title: "Move Preference"

    TelemetryChart {
        objectName: "movePreferenceChart"
        accessibleName: "Move preference"
        points: root.controller.decisionTelemetry
        firstKey: "policyPrior"
        secondKey: "visitFraction"
        firstLabel: "Policy prior"
        secondLabel: "Visit %"
        minimum: 0
        maximum: 1
        percent: true
        visible: root.controller.analysisAvailable
    }
}

