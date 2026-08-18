import QtQuick
import QtQuick.Shapes
import Style

// One caption button (minimise / maximise / close) in the custom title bar.
//
// Glyphs are drawn, not typed: the bundled typeface has no window-control
// characters, and the Windows "Segoe MDL2 Assets" font they usually come from
// is not something a cross-platform build can rely on.
Item {
    id: root

    // "minimise" | "maximise" | "restore" | "close"
    property string kind: "close"
    property bool danger: kind === "close"

    signal clicked()

    implicitWidth: 46
    implicitHeight: Theme.titleBarHeight

    readonly property color _fg: root.danger && hover.hovered ? Theme.textOnDark : Theme.text

    Rectangle {
        anchors.fill: parent
        color: !hover.hovered
               ? "transparent"
               : (root.danger ? Theme.danger : Theme.systemGray5)
    }

    // Minimise: a single rule.
    Rectangle {
        visible: root.kind === "minimise"
        anchors.centerIn: parent
        width: 10
        height: 1
        color: root._fg
    }

    // Maximise: one square.
    Rectangle {
        visible: root.kind === "maximise"
        anchors.centerIn: parent
        width: 10
        height: 10
        color: "transparent"
        border.width: 1
        border.color: root._fg
    }

    // Restore: two offset squares, the back one clipped by the front.
    Item {
        visible: root.kind === "restore"
        anchors.centerIn: parent
        width: 12
        height: 12

        Rectangle {
            x: 2; y: 0
            width: 9; height: 9
            color: "transparent"
            border.width: 1
            border.color: root._fg
        }
        Rectangle {
            x: 0; y: 3
            width: 9; height: 9
            color: Theme.surface
            border.width: 1
            border.color: root._fg
        }
    }

    // Close: an X.
    Shape {
        visible: root.kind === "close"
        anchors.centerIn: parent
        width: 10
        height: 10

        ShapePath {
            strokeColor: root._fg
            strokeWidth: 1.1
            fillColor: "transparent"
            capStyle: ShapePath.FlatCap
            startX: 0; startY: 0
            PathLine { x: 10; y: 10 }
        }
        ShapePath {
            strokeColor: root._fg
            strokeWidth: 1.1
            fillColor: "transparent"
            capStyle: ShapePath.FlatCap
            startX: 10; startY: 0
            PathLine { x: 0; y: 10 }
        }
    }

    HoverHandler { id: hover }
    TapHandler { onTapped: root.clicked() }
}
