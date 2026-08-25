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
    property string accessibleName: firstLabel + " and " + secondLabel
    property real minimum: 0
    property real maximum: 1
    property bool percent: false
    property int inspectedIndex: -1

    readonly property int displayIndex:
        inspectedIndex >= 0 && inspectedIndex < (points || []).length
        ? inspectedIndex : latestAvailableIndex()
    readonly property string firstValueText: valueText(firstKey)
    readonly property string secondValueText: valueText(secondKey)
    readonly property string firstDeltaText: deltaText(firstKey)
    readonly property string secondDeltaText: deltaText(secondKey)

    implicitHeight: 180
    Layout.fillWidth: true
    activeFocusOnTab: true

    Accessible.role: Accessible.Chart
    Accessible.name: accessibleName
    Accessible.description: firstLabel + " " + firstValueText + ", " + firstDeltaText
                            + "; " + secondLabel + " " + secondValueText + ", "
                            + secondDeltaText

    function rowAvailable(row, key) {
        return row && row.available && row[key] !== undefined && row[key] !== null
               && isFinite(Number(row[key]))
    }

    function latestAvailableIndex() {
        var rows = points || []
        for (var index = rows.length - 1; index >= 0; --index) {
            if (rowAvailable(rows[index], firstKey) || rowAvailable(rows[index], secondKey))
                return index
        }
        return -1
    }

    function previousAvailableIndex(key) {
        var rows = points || []
        for (var index = displayIndex - 1; index >= 0; --index) {
            if (rowAvailable(rows[index], key))
                return index
        }
        return -1
    }

    function exactValue(value) {
        if (!isFinite(Number(value)))
            return "—"
        return percent ? (Number(value) * 100).toFixed(2) + "%"
                       : Number(value).toFixed(2)
    }

    function valueText(key) {
        var rows = points || []
        if (displayIndex < 0 || !rowAvailable(rows[displayIndex], key))
            return "—"
        return exactValue(rows[displayIndex][key])
    }

    function deltaText(key) {
        var rows = points || []
        if (displayIndex < 0 || !rowAvailable(rows[displayIndex], key))
            return "No comparison"
        var previous = previousAvailableIndex(key)
        if (previous < 0)
            return "First value"
        var delta = Number(rows[displayIndex][key]) - Number(rows[previous][key])
        var scaled = percent ? delta * 100 : delta
        var magnitude = Math.abs(scaled).toFixed(2) + (percent ? "%" : "")
        if (scaled > 0)
            return "+" + magnitude + " up"
        if (scaled < 0)
            return "−" + magnitude + " down"
        return (percent ? "0.00%" : "0.00") + " no change"
    }

    function axisLabel(value) {
        if (percent)
            return Math.round(value * 100) + "%"
        if (value > 0)
            return "+" + value.toFixed(1)
        return value.toFixed(1)
    }

    function inspectBy(delta) {
        var rows = points || []
        if (rows.length === 0)
            return
        var start = displayIndex < 0 ? rows.length - 1 : displayIndex
        inspectedIndex = Math.max(0, Math.min(rows.length - 1, start + delta))
        chart.requestPaint()
    }

    Keys.onPressed: event => {
        if (event.key === Qt.Key_Left)
            inspectBy(-1)
        else if (event.key === Qt.Key_Right)
            inspectBy(1)
        else if (event.key === Qt.Key_Home) {
            if ((points || []).length > 0)
                inspectedIndex = 0
            chart.requestPaint()
        } else if (event.key === Qt.Key_End) {
            inspectedIndex = latestAvailableIndex()
            chart.requestPaint()
        } else {
            return
        }
        event.accepted = true
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.spacing

            Repeater {
                model: [
                    { label: root.firstLabel, color: Theme.accent,
                      value: root.firstValueText, delta: root.firstDeltaText },
                    { label: root.secondLabel, color: Theme.systemOrange,
                      value: root.secondValueText, delta: root.secondDeltaText }
                ]

                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: 42
                    radius: Theme.radiusSmall
                    color: Theme.surfaceAlt
                    border.width: 1
                    border.color: Theme.border

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 6
                        spacing: 6

                        Rectangle {
                            width: 3
                            Layout.fillHeight: true
                            radius: 2
                            color: parent.parent.modelData.color
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 0
                            Text {
                                Layout.fillWidth: true
                                text: parent.parent.parent.modelData.label
                                color: Theme.textMuted
                                font.family: Theme.fontFamily
                                font.pixelSize: Theme.fontTiny
                                elide: Text.ElideRight
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                Text {
                                    text: parent.parent.parent.parent.modelData.value
                                    color: Theme.text
                                    font.family: Theme.fontFamily
                                    font.pixelSize: Theme.fontBody
                                    font.weight: Theme.weightBold
                                }
                                Item { Layout.fillWidth: true }
                                Text {
                                    text: parent.parent.parent.parent.modelData.delta
                                    color: text.indexOf("up") >= 0 ? Theme.success
                                         : text.indexOf("down") >= 0 ? Theme.danger
                                         : Theme.textFaint
                                    font.family: Theme.fontFamily
                                    font.pixelSize: Theme.fontTiny
                                }
                            }
                        }
                    }
                }
            }
        }

        Item {
            id: plot
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 104

            Canvas {
                id: chart
                anchors.fill: parent

                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.clearRect(0, 0, width, height)
                    var left = 42
                    var right = width - 5
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
                        var axisValue = root.maximum - ratio * (root.maximum - root.minimum)
                        ctx.fillText(root.axisLabel(axisValue), left - 5, y)
                    }

                    var rows = root.points || []
                    if (rows.length === 0)
                        return
                    var minPly = rows[0].ply
                    var maxPly = rows[rows.length - 1].ply
                    function xFor(index) {
                        return rows.length === 1 ? left + plotWidth / 2
                                                 : left + index * plotWidth / (rows.length - 1)
                    }
                    function yFor(value) {
                        return bottom - (value - root.minimum) * plotHeight
                                        / (root.maximum - root.minimum)
                    }
                    function drawSeries(key, color) {
                        ctx.strokeStyle = color
                        ctx.lineWidth = 2
                        ctx.beginPath()
                        var drawing = false
                        for (var index = 0; index < rows.length; ++index) {
                            var row = rows[index]
                            if (!root.rowAvailable(row, key)) {
                                drawing = false
                                continue
                            }
                            var x = xFor(index)
                            var y = yFor(Number(row[key]))
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

                    if (root.inspectedIndex >= 0 && root.inspectedIndex < rows.length) {
                        var selectedX = xFor(root.inspectedIndex)
                        ctx.strokeStyle = Theme.textMuted
                        ctx.lineWidth = 1
                        ctx.beginPath()
                        ctx.moveTo(selectedX, top)
                        ctx.lineTo(selectedX, bottom)
                        ctx.stroke()
                        var keys = [root.firstKey, root.secondKey]
                        var colors = [Theme.accent, Theme.systemOrange]
                        for (var marker = 0; marker < keys.length; ++marker) {
                            if (!root.rowAvailable(rows[root.inspectedIndex], keys[marker]))
                                continue
                            ctx.fillStyle = colors[marker]
                            ctx.beginPath()
                            ctx.arc(selectedX,
                                    yFor(Number(rows[root.inspectedIndex][keys[marker]])),
                                    3.5, 0, Math.PI * 2)
                            ctx.fill()
                        }
                    }

                    ctx.fillStyle = Theme.textFaint
                    ctx.textBaseline = "bottom"
                    ctx.textAlign = "left"
                    ctx.fillText(String(minPly), left, height)
                    ctx.textAlign = "right"
                    ctx.fillText(String(maxPly), right, height)
                }
            }

            MouseArea {
                id: hoverArea
                anchors.fill: parent
                hoverEnabled: true
                acceptedButtons: Qt.NoButton

                onPositionChanged: mouse => {
                    var rows = root.points || []
                    if (rows.length === 0)
                        return
                    var left = 42
                    var right = width - 5
                    var ratio = Math.max(0, Math.min(1, (mouse.x - left)
                                                    / Math.max(1, right - left)))
                    root.inspectedIndex = Math.round(ratio * (rows.length - 1))
                    chart.requestPaint()
                }
                onExited: {
                    root.inspectedIndex = -1
                    chart.requestPaint()
                }
            }

            Rectangle {
                visible: hoverArea.containsMouse && root.displayIndex >= 0
                z: 2
                x: Math.max(4, Math.min(plot.width - width - 4, hoverArea.mouseX + 10))
                y: 4
                implicitWidth: tooltipText.implicitWidth + 16
                implicitHeight: tooltipText.implicitHeight + 12
                radius: Theme.radiusSmall
                color: Theme.surface
                border.width: 1
                border.color: Theme.borderStrong

                Text {
                    id: tooltipText
                    anchors.centerIn: parent
                    text: root.displayIndex < 0 ? "" :
                          "Turn " + String((root.points[root.displayIndex] || {}).ply)
                          + "\n" + root.firstLabel + ": " + root.firstValueText
                          + "\n" + root.secondLabel + ": " + root.secondValueText
                    color: Theme.text
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTiny
                }
            }

            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.width: root.activeFocus ? 1 : 0
                border.color: Theme.selection
                radius: Theme.radiusSmall
            }
        }
    }

    onPointsChanged: {
        inspectedIndex = -1
        chart.requestPaint()
    }
    onMinimumChanged: chart.requestPaint()
    onMaximumChanged: chart.requestPaint()
}
