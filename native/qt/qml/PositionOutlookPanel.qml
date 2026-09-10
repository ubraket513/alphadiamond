import QtQuick
import QtQuick.Layouts
import Style

PanelSection {
    id: root
    objectName: "positionOutlookPanel"
    required property var controller
    title: "Position Outlook"
    subtitle: root.outlookPlayerId === 0 ? "No AI player"
              : "AI · P" + root.outlookPlayerId + (root.controller.playerCount === 3
                ? " · Placement utility" : " · Estimated outlook")

    // Keep one stable AI perspective; never connect different players' utilities.
    readonly property int outlookPlayerId: Number((controller.aiSeats || [])[0] || 0)
    readonly property var outlookPoints: {
        var rows = controller.positionTelemetry || []
        var result = []
        for (var i = 0; i < rows.length; ++i) {
            var row = rows[i]
            if (Number(row.playerId) !== outlookPlayerId)
                continue
            var point = Object.assign({}, row)
            if (controller.playerCount === 2
                    && Number(row.perspectivePlayerId) !== outlookPlayerId && row.available) {
                point.nnValue = -Number(row.nnValue)
                point.mctsValue = -Number(row.mctsValue)
                point.nnEstimate = 1 - Number(row.nnEstimate)
                point.mctsEstimate = 1 - Number(row.mctsEstimate)
            }
            result.push(point)
        }
        return result
    }

    TelemetryChart {
        objectName: "positionOutlookChart"
        accessibleName: "Position outlook"
        points: root.outlookPoints

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
