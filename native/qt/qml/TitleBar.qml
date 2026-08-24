import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Window
import Style

// Custom window chrome, replacing the native title bar.
//
// Layout mirrors the reference console: a panel-toggle button at the far left,
// then a flat menu row (File / Edit / View / Window / Help), a draggable
// spacer carrying the window title, and the caption buttons hard right.
//
// The window is frameless, so this item owns everything the OS would normally
// provide: dragging (startSystemMove), double-click to maximise, and the
// minimise / maximise / close buttons. Edge resizing lives in Main.qml, which
// is the only item that spans the whole window.
Rectangle {
    id: root
    objectName: "titleBar"

    required property var controller
    required property var window

    property alias panelVisible: panelToggle.on

    signal newGameRequested()
    signal saveRequested()
    signal loadRequested()
    signal soundsRequested()
    signal historyRequested()
    signal aboutRequested()

    readonly property bool maximised: window.visibility === Window.Maximized
                                      || window.visibility === Window.FullScreen

    readonly property bool nativeHover:
        (typeof nativeChrome !== "undefined" && nativeChrome)
        ? nativeChrome.maximiseHovered : false

    // The OS swallows the click once it owns the button, so the toggle arrives
    // as a signal instead of a TapHandler.
    Connections {
        target: (typeof nativeChrome !== "undefined") ? nativeChrome : null
        ignoreUnknownSignals: true
        function onMaximiseClicked() { root.toggleMaximised() }
    }

    // The button rides the right edge, so its position changes with the window.
    Connections {
        target: root.window
        function onWidthChanged() { maximiseButton.reportRect() }
    }

    implicitHeight: Theme.titleBarHeight
    color: Theme.surface

    // Hairline under the bar, as in the reference.
    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.border
    }

    // Dragging anywhere that is not a control moves the window; double-click
    // toggles maximise, matching a native title bar.
    TapHandler {
        onDoubleTapped: root.toggleMaximised()
        gesturePolicy: TapHandler.DragThreshold
    }
    DragHandler {
        target: null
        grabPermissions: PointerHandler.CanTakeOverFromAnything
        onActiveChanged: if (active) root.window.startSystemMove()
    }

    function toggleMaximised() {
        window.visibility = root.maximised ? Window.Windowed : Window.Maximized
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // -- panel toggle (the reference's hamburger) ---------------------
        Item {
            id: panelToggle
            property bool on: true

            implicitWidth: 44
            implicitHeight: Theme.titleBarHeight

            Rectangle {
                anchors.centerIn: parent
                width: 30; height: 26
                radius: Theme.radiusSmall
                color: burgerHover.hovered ? Theme.systemGray6 : "transparent"
            }

            Icon {
                anchors.centerIn: parent
                name: "msc.menu"
                size: 16
                color: Theme.text
            }

            HoverHandler { id: burgerHover; cursorShape: Qt.PointingHandCursor }
            TapHandler { onTapped: panelToggle.on = !panelToggle.on }
        }

        // -- menus --------------------------------------------------------
        TitleMenu {
            text: "File"
            model: [
                { label: "New Game",  shortcut: "Ctrl+N", action: "new" },
                { label: "Save…",     shortcut: "Ctrl+S", action: "save" },
                { label: "Load…",     shortcut: "Ctrl+O", action: "load" },
                { separator: true },
                { label: "Exit",      shortcut: "Alt+F4", action: "exit" }
            ]
            onTriggered: function (action) {
                if (action === "new")       root.newGameRequested()
                else if (action === "save") root.saveRequested()
                else if (action === "load") root.loadRequested()
                else if (action === "exit") root.window.close()
            }
        }

        TitleMenu {
            text: "Edit"
            model: [
                {
                    label: "Undo move", shortcut: "Ctrl+Z", action: "undo",
                    enabled: root.controller.canUndo
                },
                { separator: true },
                {
                    label: "Confirm move", shortcut: "Enter", action: "confirm",
                    enabled: root.controller.canConfirm
                },
                {
                    label: "Cancel", shortcut: "Esc", action: "cancel",
                    enabled: root.controller.canCancel
                }
            ]
            onTriggered: function (action) {
                if (action === "undo")         root.controller.undoLastMove()
                else if (action === "confirm") root.controller.confirmProposal()
                else if (action === "cancel")  root.controller.cancelProposal()
            }
        }

        TitleMenu {
            text: "View"
            model: [
                {
                    label: root.panelVisible ? "Hide side panel" : "Show side panel",
                    action: "panel"
                },
                { label: "History", action: "history" },
                { separator: true },
                // Muting lives in the Sounds dialog alongside the volume; a
                // second entry here would be a second place to change one
                // setting.
                { label: "Sounds…", action: "sounds" }
            ]
            onTriggered: function (action) {
                if (action === "panel")
                    panelToggle.on = !panelToggle.on
                else if (action === "history")
                    root.historyRequested()
                else if (action === "sounds")
                    root.soundsRequested()
            }
        }

        TitleMenu {
            text: "Window"
            model: [
                { label: "Minimise", action: "min" },
                {
                    label: root.maximised ? "Restore" : "Maximise",
                    action: "max"
                },
                { separator: true },
                { label: "Close", shortcut: "Alt+F4", action: "close" }
            ]
            onTriggered: function (action) {
                if (action === "min")        root.window.showMinimized()
                else if (action === "max")   root.toggleMaximised()
                else if (action === "close") root.window.close()
            }
        }

        TitleMenu {
            text: "Help"
            model: [{ label: "About Diamond", action: "about" }]
            onTriggered: function (action) {
                if (action === "about") root.aboutRequested()
            }
        }

        // -- draggable centre, carrying the window title ------------------
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Text {
                anchors.centerIn: parent
                text: "Diamond  ·  " + root.controller.gameLabel + "  ·  "
                      + root.controller.playerCount + "-player"
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                elide: Text.ElideRight
                width: Math.min(implicitWidth, parent.width - Theme.spacingLarge)
                horizontalAlignment: Text.AlignHCenter
            }
        }

        // -- caption buttons ----------------------------------------------
        WindowButton {
            kind: "minimise"
            onClicked: root.window.showMinimized()
        }
        WindowButton {
            id: maximiseButton
            kind: root.maximised ? "restore" : "maximise"
            onClicked: root.toggleMaximised()

            // Snap Layouts: Windows only offers the flyout to a window whose
            // hit test claims this button, so the filter has to know where it
            // is. Reported in the window's logical coordinates.
            externalHover: root.nativeHover

            function reportRect() {
                if (typeof nativeChrome === "undefined" || !nativeChrome)
                    return
                var p = mapToItem(null, 0, 0)
                nativeChrome.setMaximiseButtonRect(p.x, p.y, width, height)
            }

            onWidthChanged: reportRect()
            onHeightChanged: reportRect()
            onXChanged: reportRect()
            Component.onCompleted: reportRect()
        }
        WindowButton {
            kind: "close"
            onClicked: root.window.close()
        }
    }
}
