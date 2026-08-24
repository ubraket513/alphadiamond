import QtQuick
import Style

// One caption button (minimise / maximise / close) in the custom title bar.
//
// The glyphs are Microsoft's own Codicons, shipped by QtAwesome — the same
// shapes the shell draws, without depending on "Segoe MDL2 Assets" being
// present, which a cross-platform build cannot assume.
Item {
    id: root

    // "minimise" | "maximise" | "restore" | "close"
    property string kind: "close"
    property bool danger: kind === "close"

    // Windows owns the pointer once the hit test reports HTMAXBUTTON, so Qt
    // stops seeing hover over the maximise button. The filter reports it here.
    property bool externalHover: false

    readonly property bool hovered: hover.hovered || externalHover

    signal clicked()

    implicitWidth: 46
    implicitHeight: Theme.titleBarHeight

    readonly property string _glyph: kind === "minimise" ? "msc.chrome-minimize"
                                   : kind === "maximise" ? "msc.chrome-maximize"
                                   : kind === "restore"  ? "msc.chrome-restore"
                                                         : "msc.chrome-close"

    Rectangle {
        anchors.fill: parent
        color: !root.hovered
               ? "transparent"
               : (root.danger ? Theme.danger : Theme.systemGray5)
    }

    Icon {
        anchors.centerIn: parent
        name: root._glyph
        size: 14
        color: root.danger && root.hovered ? Theme.textOnDark : Theme.text
    }

    HoverHandler { id: hover }
    TapHandler { onTapped: root.clicked() }
}
