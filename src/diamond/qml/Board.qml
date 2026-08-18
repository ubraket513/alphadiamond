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
        campCanvas.requestPaint()
        latticeCanvas.requestPaint()
    }

    onUnitScaleChanged: {
        campCanvas.requestPaint()
        latticeCanvas.requestPaint()
        pathCanvas.requestPaint()
    }

    // A new match can change the seat colours, so the wash must be repainted.
    Connections {
        target: root.geo
        function onChanged() {
            root._camps = root.geo.camps()
            campCanvas.requestPaint()
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.boardBackground
        border.width: 1
        border.color: Theme.border
        radius: Theme.radiusLarge
    }

    // Camp regions.
    //
    // Drawn on their own canvas at full opacity and composited once via the
    // item's `opacity`. Painting them translucent individually would double the
    // alpha wherever two camps overlap -- and adjacent camps always overlap,
    // because they share a hexagon corner hole -- leaving a hard dark notch at
    // every junction. One translucent layer has no seam.
    //
    // Corners are rounded rather than mitred: sharp vertices read as diagram
    // furniture, and the rounding also pulls neighbouring camps apart at the
    // corner they share, so the junction resolves into a soft gap instead of a
    // collision.
    Canvas {
        id: campCanvas
        anchors.fill: parent
        antialiasing: true
        renderStrategy: Canvas.Cooperative
        opacity: Theme.campFillAlpha

        // Convex hull (monotone chain) of the camp's hole centres.
        function hull(points) {
            if (points.length < 3)
                return points.slice()
            const pts = points.slice().sort(function (a, b) {
                return a.x === b.x ? a.y - b.y : a.x - b.x
            })
            const cross = function (o, a, b) {
                return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
            }
            const lower = []
            for (let i = 0; i < pts.length; ++i) {
                while (lower.length >= 2
                       && cross(lower[lower.length - 2], lower[lower.length - 1], pts[i]) <= 0)
                    lower.pop()
                lower.push(pts[i])
            }
            const upper = []
            for (let i = pts.length - 1; i >= 0; --i) {
                while (upper.length >= 2
                       && cross(upper[upper.length - 2], upper[upper.length - 1], pts[i]) <= 0)
                    upper.pop()
                upper.push(pts[i])
            }
            lower.pop()
            upper.pop()
            return lower.concat(upper)
        }

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)

            // The region is the camp's hull grown outward by this much. Filling
            // the hull and stroking its outline with a round-joined pen of
            // twice the width is a Minkowski sum with a disc: a smooth rounded
            // shape offset from the real hole set, with no vertices to mitre
            // and none of the scalloping a union of separate discs would leave.
            const r = Theme.campDiscRadius * root.unitScale

            ctx.lineJoin = "round"
            ctx.lineCap = "round"
            ctx.lineWidth = 2 * r

            for (let c = 0; c < root._camps.length; ++c) {
                const camp = root._camps[c]
                const holes = camp.holes
                if (!holes || holes.length === 0)
                    continue

                const screen = []
                for (let i = 0; i < holes.length; ++i)
                    screen.push({ x: root.mapX(holes[i].x), y: root.mapY(holes[i].y) })

                const outline = campCanvas.hull(screen)
                const paint = camp.inPlay ? camp.color : Theme.campNeutral

                ctx.beginPath()
                ctx.moveTo(outline[0].x, outline[0].y)
                for (let i = 1; i < outline.length; ++i)
                    ctx.lineTo(outline[i].x, outline[i].y)
                ctx.closePath()

                ctx.fillStyle = paint
                ctx.strokeStyle = paint
                ctx.fill()
                ctx.stroke()
            }
        }
    }

    // Lattice lines, above the camp wash so the grid stays continuous across
    // region boundaries.
    Canvas {
        id: latticeCanvas
        anchors.fill: parent
        antialiasing: true
        renderStrategy: Canvas.Cooperative

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)

            ctx.strokeStyle = Theme.lattice
            ctx.lineWidth = Theme.latticeWidth
            ctx.lineCap = "round"
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
