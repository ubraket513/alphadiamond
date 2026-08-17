import QtQuick
import Style

// One playable hole. Rendering + hit testing only; legality is decided in Python.
Item {
    id: root

    required property int positionId
    required property bool isSelected
    required property bool isLegalStep
    required property bool isLegalJump
    required property bool isPathNode
    required property int  pathIndex
    required property bool isLastMoveSource
    required property bool isLastMoveDest
    required property bool isProposalSource
    required property bool isProposalDest

    property real unitScale: 1
    property bool interactive: true

    signal clicked(int positionId)

    readonly property bool isLegal: isLegalStep || isLegalJump
    readonly property real holeRadius: unitScale * Theme.holeRatio

    // Base node: a solid black circle, per the reference board art.
    Rectangle {
        id: node
        anchors.centerIn: parent
        width: root.holeRadius * 2
        height: width
        radius: width / 2
        color: Theme.hole
        antialiasing: true
    }

    // Last-move memory: two thin neutral rings, deliberately quiet.
    Rectangle {
        anchors.centerIn: parent
        visible: root.isLastMoveSource || root.isLastMoveDest
        width: root.holeRadius * 3.2
        height: width
        radius: width / 2
        color: "transparent"
        border.width: Math.max(1, root.unitScale * 0.035)
        border.color: Theme.lastMove
        opacity: root.isLastMoveDest ? 0.9 : 0.5
        antialiasing: true
    }

    // Legal destination marker. Step and jump are visually distinguished.
    Rectangle {
        anchors.centerIn: parent
        visible: root.isLegal
        width: root.holeRadius * (root.isLegalJump ? 3.6 : 2.9)
        height: width
        radius: width / 2
        color: "transparent"
        border.width: Math.max(1.5, root.unitScale * (root.isLegalJump ? 0.075 : 0.05))
        border.color: root.isLegalJump ? Theme.legalJump : Theme.legalStep
        antialiasing: true
    }

    // Selected piece.
    Rectangle {
        anchors.centerIn: parent
        visible: root.isSelected
        width: root.holeRadius * 4.4
        height: width
        radius: width / 2
        color: "transparent"
        border.width: Math.max(1.5, root.unitScale * 0.06)
        border.color: Theme.selection
        antialiasing: true
    }

    // Proposal endpoints and numbered intermediate jump markers.
    Rectangle {
        anchors.centerIn: parent
        visible: root.isProposalSource || root.isProposalDest
        width: root.holeRadius * 4.4
        height: width
        radius: width / 2
        color: "transparent"
        border.width: Math.max(1.5, root.unitScale * 0.07)
        border.color: Theme.proposal
        antialiasing: true
    }

    Rectangle {
        anchors.centerIn: parent
        visible: root.isPathNode && root.pathIndex > 0 && !root.isProposalDest
        width: root.holeRadius * 2.6
        height: width
        radius: width / 2
        color: Theme.surface
        border.width: 1
        border.color: Theme.proposal
        antialiasing: true

        Text {
            anchors.centerIn: parent
            text: root.pathIndex
            color: Theme.proposal
            font.family: Theme.fontFamily
            font.pixelSize: Math.max(8, parent.width * 0.68)
            font.weight: Theme.weightBold
        }
    }

    MouseArea {
        anchors.centerIn: parent
        width: Math.max(20, root.unitScale * 0.8)
        height: width
        enabled: root.interactive
        cursorShape: root.interactive ? Qt.PointingHandCursor : Qt.ArrowCursor
        hoverEnabled: true
        onClicked: root.clicked(root.positionId)

        Rectangle {
            anchors.centerIn: parent
            visible: parent.containsMouse && root.interactive
            width: root.holeRadius * 3.0
            height: width
            radius: width / 2
            color: "transparent"
            border.width: 1
            border.color: Theme.borderStrong
            antialiasing: true
        }
    }
}
