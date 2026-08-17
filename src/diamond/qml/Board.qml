import QtQuick
import Style

// The board is drawn procedurally from logical lattice coordinates supplied by
// the engine (controller.geometry). No pixel position is hard-coded and no
// pre-rendered board image is used, so it stays sharp at any window size.
Item {
    id: root

    required property var controller

    readonly property var geo: controller.geometry

    property var _bounds: ({ minX: -1, minY: -1, maxX: 1, maxY: 1 })
    property var _edges: []
    property var _camps: []
    property var _holePoints: []

    // Scale so the whole star fits, preserving lattice geometry and spacing.
    readonly property real _spanX: _bounds.maxX - _bounds.minX
    readonly property real _spanY: _bounds.maxY - _bounds.minY
    readonly property real unitScale: Math.max(1, Math.min(
        (width  - 2 * Theme.boardPadding) / _spanX,
        (height - 2 * Theme.boardPadding) / _spanY))
    readonly property real _originX: (width  - _spanX * unitScale) / 2 - _bounds.minX * unitScale
    readonly property real _originY: (height - _spanY * unitScale) / 2 - _bounds.minY * unitScale

    function mapX(ux) { return _originX + ux * unitScale }
    function mapY(uy) { return _originY + uy * unitScale }

    Component.onCompleted: {
        _bounds = geo.bounds()
        _edges = geo.edges()
        _camps = geo.camps()
        _holePoints = geo.holes()
        latticeCanvas.requestPaint()
    }

    onUnitScaleChanged: { latticeCanvas.requestPaint(); pathCanvas.requestPaint() }

    Rectangle {
        anchors.fill: parent
        color: Theme.boardBackground
        border.width: 1
        border.color: Theme.border
        radius: Theme.radiusMedium
    }

    // Camp fills + lattice lines. Flat colour, thin strokes, no shading.
    Canvas {
        id: latticeCanvas
        anchors.fill: parent
        antialiasing: true
        renderStrategy: Canvas.Cooperative

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)

            for (let c = 0; c < root._camps.length; ++c) {
                const camp = root._camps[c]
                const pts = camp.points
                if (!pts || pts.length < 3)
                    continue
                // Expand the triangle slightly so the fill reaches past the
                // outermost node centres, as in the reference art.
                const cx = (pts[0].x + pts[1].x + pts[2].x) / 3
                const cy = (pts[0].y + pts[1].y + pts[2].y) / 3
                const grow = 1.18
                ctx.beginPath()
                for (let i = 0; i < 3; ++i) {
                    const px = root.mapX(cx + (pts[i].x - cx) * grow)
                    const py = root.mapY(cy + (pts[i].y - cy) * grow)
                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
                }
                ctx.closePath()
                const tint = Qt.color(camp.color)
                ctx.fillStyle = Qt.rgba(tint.r, tint.g, tint.b, Theme.campFillAlpha)
                ctx.fill()
            }

            ctx.strokeStyle = Theme.lattice
            ctx.lineWidth = Theme.latticeWidth
            ctx.beginPath()
            for (let e = 0; e < root._edges.length; ++e) {
                const edge = root._edges[e]
                ctx.moveTo(root.mapX(edge.x1), root.mapY(edge.y1))
                ctx.lineTo(root.mapX(edge.x2), root.mapY(edge.y2))
            }
            ctx.stroke()
        }
    }

    // Proposed / AI move path drawn above the lattice, below the pieces.
    Canvas {
        id: pathCanvas
        anchors.fill: parent
        antialiasing: true
        renderStrategy: Canvas.Cooperative

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)

            const ids = root.controller.proposalPathIds
            if (!ids || ids.length < 2)
                return

            ctx.strokeStyle = Theme.pathLine
            ctx.lineWidth = Math.max(2, root.unitScale * 0.09)
            ctx.lineCap = "round"
            ctx.lineJoin = "round"
            ctx.beginPath()
            for (let i = 0; i < ids.length; ++i) {
                const p = root._holePoints[ids[i]]
                const px = root.mapX(p.x)
                const py = root.mapY(p.y)
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
            }
            ctx.stroke()
        }
    }

    Connections {
        target: root.controller
        function onChanged() { pathCanvas.requestPaint() }
    }

    // Model roles are injected into the delegates' required properties by name;
    // the flag roles are already declared as required in Hole.qml / Piece.qml.
    Repeater {
        model: root.controller.boardModel
        delegate: Hole {
            required property real unitX
            required property real unitY

            unitScale: root.unitScale
            interactive: root.controller.canSelect
            width: root.unitScale
            height: root.unitScale
            x: root.mapX(unitX) - width / 2
            y: root.mapY(unitY) - height / 2

            onClicked: function (pid) { root.controller.selectPosition(pid) }
        }
    }

    Repeater {
        model: root.controller.pieceModel
        delegate: Piece {
            required property real   unitX
            required property real   unitY
            required property string color
            required property int    positionId

            pieceColor: color
            unitScale: root.unitScale
            active: playerId === root.controller.currentPlayerId
                    && !root.controller.isGameOver
            highlighted: positionId === root.controller.selectedPosition
            width: root.unitScale
            height: root.unitScale
            x: root.mapX(unitX) - width / 2
            y: root.mapY(unitY) - height / 2
        }
    }
}
