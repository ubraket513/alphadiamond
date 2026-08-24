import QtQuick
import QtQuick.Layouts
import Style

Item {
    id: root

    property var points: []
    property string firstKey: ""
    property string secondKey: ""
    property string firstLabel: ""
    property string secondLabel: ""
    property real minimum: 0
    property real maximum: 1
    property bool percent: false

    implicitHeight: 116
    Layout.fillWidth: true

    function axisLabel(value) {
        if (percent)
            return Math.round(value * 100) + "%"
        if (value > 0)
            return "+" + value.toFixed(1)
        return value.toFixed(1)
    }

    RowLayout {
        id: legend
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        height: 16
        spacing: Theme.spacing

        Repeater {
            model: [
                { label: root.firstLabel, color: Theme.accent },
                { label: root.secondLabel, color: Theme.systemOrange }
            ]
            delegate: RowLayout {
                required property var modelData
                spacing: 4
                Rectangle { width: 12; height: 2; color: parent.modelData.color }
                Text {
                    text: parent.modelData.label
                    color: Theme.textMuted
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                }
            }
        }
        Item { Layout.fillWidth: true }
    }

    Canvas {
        id: canvas
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: legend.bottom
        anchors.bottom: parent.bottom

        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            var left = 34
            var right = width - 4
            var top = 5
            var bottom = height - 18
            var plotWidth = Math.max(1, right - left)
            var plotHeight = Math.max(1, bottom - top)

            ctx.lineWidth = 1
            ctx.strokeStyle = Theme.border
            ctx.fillStyle = Theme.textFaint
            ctx.font = Theme.fontTiny + "px " + Theme.fontFamily
            ctx.textAlign = "right"
            ctx.textBaseline = "middle"
            for (var grid = 0; grid <= 2; ++grid) {
                var ratio = grid / 2
                var y = top + ratio * plotHeight
                ctx.beginPath()
                ctx.moveTo(left, y)
                ctx.lineTo(right, y)
                ctx.stroke()
                var value = root.maximum - ratio * (root.maximum - root.minimum)
                ctx.fillText(root.axisLabel(value), left - 5, y)
            }

            var rows = root.points || []
            if (rows.length === 0)
                return
            var minPly = rows[0].ply
            var maxPly = rows[rows.length - 1].ply
            function xFor(ply) {
                return minPly === maxPly ? left + plotWidth / 2
                                         : left + (ply - minPly) * plotWidth / (maxPly - minPly)
            }
            function yFor(value) {
                return bottom - (value - root.minimum) * plotHeight / (root.maximum - root.minimum)
            }
            function drawSeries(key, color) {
                ctx.strokeStyle = color
                ctx.lineWidth = 1.5
                ctx.beginPath()
                var drawing = false
                for (var index = 0; index < rows.length; ++index) {
                    var row = rows[index]
                    var value = row[key]
                    if (!row.available || value === undefined || value === null) {
                        drawing = false
                        continue
                    }
                    var x = xFor(row.ply)
                    var y = yFor(value)
                    if (!drawing) {
                        ctx.moveTo(x, y)
                        drawing = true
                    } else {
                        ctx.lineTo(x, y)
                    }
                }
                ctx.stroke()
            }
            drawSeries(root.firstKey, Theme.accent)
            drawSeries(root.secondKey, Theme.systemOrange)

            ctx.fillStyle = Theme.textFaint
            ctx.textBaseline = "bottom"
            ctx.textAlign = "left"
            ctx.fillText(String(minPly), left, height)
            ctx.textAlign = "right"
            ctx.fillText(String(maxPly), right, height)
        }
    }

    onPointsChanged: canvas.requestPaint()
    onMinimumChanged: canvas.requestPaint()
    onMaximumChanged: canvas.requestPaint()
}

