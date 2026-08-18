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
    //
    // `_bounds` covers hole *centres*, but a socket reaches a further
    // `socketRatio` units beyond the outermost one. Adding that overhang to the
    // span is what keeps the edge sockets inside the panel rather than clipping
    // against it. The camp triangles stop at hole centres, so they add nothing.
    readonly property real _overhang: 2 * Theme.socketRatio
    readonly property real _spanX: (_bounds.maxX - _bounds.minX) + _overhang
    readonly property real _spanY: (_bounds.maxY - _bounds.minY) + _overhang
    readonly property real unitScale: Math.max(1, Math.min(
        (width  - 2 * Theme.boardPadding) / _spanX,
        (height - 2 * Theme.boardPadding) / _spanY))
    readonly property real _originX: (width  - _spanX * unitScale) / 2
                                     + (_overhang / 2 - _bounds.minX) * unitScale
    readonly property real _originY: (height - _spanY * unitScale) / 2
                                     + (_overhang / 2 - _bounds.minY) * unitScale

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

    // Camp regions: a sharp triangle per camp, vertices sitting exactly on the
    // camp's three corner holes.
    //
    // Because adjacent camps share precisely one hexagon-corner hole, their
    // triangles meet at that single vertex and never overlap by area — so the
    // junction needs no special treatment and no colour has to win over
    // another. That is what lets these be plain mitred triangles rather than
    // the offset hulls they used to be.
    //
    // The layer sits at the very back and is composited once through the item's
    // own `opacity`; everything else on the board is drawn over it.
    // Stacking order is load-bearing here, so every layer states its own z:
    //
    //   0 camp triangles   1 lattice   2 holes   3 move path   4 pieces   5 hop numbers
    //
    // The path runs *over* empty holes and *under* pieces: hopping over a piece
    // should read as passing behind it, while a landing on an empty hole should
    // stay visible. The hop numbers sit above the path so the line cannot cut
    // through them.
    Canvas {
        id: campCanvas
        z: 0
        anchors.fill: parent
        antialiasing: true
        renderStrategy: Canvas.Cooperative
        opacity: Theme.campFillAlpha

        onPaint: {
            const ctx = getContext("2d")
            ctx.reset()
            ctx.clearRect(0, 0, width, height)

            for (let c = 0; c < root._camps.length; ++c) {
                const camp = root._camps[c]
                const pts = camp.points
                if (!pts || pts.length < 3)
                    continue

                ctx.beginPath()
                for (let i = 0; i < 3; ++i) {
                    const px = root.mapX(pts[i].x)
                    const py = root.mapY(pts[i].y)
                    if (i === 0) ctx.moveTo(px, py)
                    else ctx.lineTo(px, py)
                }
                ctx.closePath()
                ctx.fillStyle = camp.inPlay ? camp.color : Theme.campNeutral
                ctx.fill()
            }
        }
    }

    // Lattice lines, above the camp wash so the grid stays continuous across
    // region boundaries.
    Canvas {
        id: latticeCanvas
        z: 1
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

            // Each segment is trimmed back by a socket radius at both ends, so
            // it runs *between* two sockets instead of straight through them.
            // Untrimmed, every empty socket reads as a hole with an X drawn
            // across it.
            const trim = Theme.socketRatio * root.unitScale

            ctx.beginPath()
            for (let e = 0; e < root._edges.length; ++e) {
                const edge = root._edges[e]
                const x1 = root.mapX(edge.x1)
                const y1 = root.mapY(edge.y1)
                const x2 = root.mapX(edge.x2)
                const y2 = root.mapY(edge.y2)
                const dx = x2 - x1
                const dy = y2 - y1
                const len = Math.sqrt(dx * dx + dy * dy)
                if (len <= 2 * trim)
                    continue          // sockets already meet; no line to draw
                const ux = dx / len
                const uy = dy / len
                ctx.moveTo(x1 + ux * trim, y1 + uy * trim)
                ctx.lineTo(x2 - ux * trim, y2 - uy * trim)
            }
            ctx.stroke()
        }
    }

    // Proposed / AI move path drawn above the lattice, below the pieces.
    Canvas {
        id: pathCanvas
        z: 3
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

            z: 2
            unitScale: root.unitScale
            interactive: root.controller.canSelect
            width: root.unitScale
            height: root.unitScale
            x: root.mapX(unitX) - width / 2
            y: root.mapY(unitY) - height / 2

            onClicked: function (pid) { root.controller.selectPosition(pid) }
        }
    }

    // Numbered landings of a jump chain, so the order can be read. Drawn above
    // the path line rather than inside Hole.qml, which sits below it.
    // The destination is deliberately left unnumbered — it is already the end
    // of the line, and the ghost fill marks it.
    Repeater {
        model: root.controller.boardModel

        delegate: Item {
            required property real unitX
            required property real unitY
            required property bool isPathNode
            required property int  pathIndex
            required property bool isProposalDest

            z: 5
            visible: isPathNode && pathIndex > 0 && !isProposalDest
            width: root.unitScale
            height: root.unitScale
            x: root.mapX(unitX) - width / 2
            y: root.mapY(unitY) - height / 2

            Rectangle {
                anchors.centerIn: parent
                width: root.unitScale * Theme.socketRatio * 1.05
                height: width
                radius: width / 2
                color: Theme.surface
                border.width: 1
                border.color: Theme.proposal
                antialiasing: true

                Text {
                    anchors.centerIn: parent
                    text: parent.parent.pathIndex
                    color: Theme.proposal
                    font.family: Theme.fontFamily
                    font.pixelSize: Math.max(8, parent.width * 0.7)
                    font.weight: Theme.weightBold
                }
            }
        }
    }

    Repeater {
        model: root.controller.pieceModel
        delegate: Piece {
            required property string color
            required property int    positionId

            z: 4
            pieceColor: color
            // The mapping is handed over rather than applied here, so the piece
            // can ease its lattice position while tracking board rescaling
            // instantly. See the note in Piece.qml.
            originX: root._originX
            originY: root._originY
            unitScale: root.unitScale
            highlighted: positionId === root.controller.selectedPosition
            width: root.unitScale
            height: root.unitScale
        }
    }
}
