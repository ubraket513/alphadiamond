import QtQuick
import QtQuick.Controls.Basic
import Style

// The app's scroll bar.
//
// Like every other control here, it paints its own contentItem and background
// rather than inheriting the Basic style's. That style's default handle picks
// its colour from the platform palette (`palette.mid` / `palette.dark`) and
// branches on `Qt.styleHints.accessibility.contrastPreference` — neither of
// which is under this app's control, and both of which have to resolve on
// whatever platform and colour scheme the operator happens to be running.
// Styling it removes that dependency and matches the rest of the chrome.
ScrollBar {
    id: control

    policy: ScrollBar.AsNeeded
    padding: 2

    contentItem: Rectangle {
        implicitWidth: control.interactive ? 6 : 3
        implicitHeight: control.interactive ? 6 : 3
        radius: width / 2
        color: control.pressed ? Theme.textMuted
                               : (control.hovered ? Theme.systemGray : Theme.systemGray3)
        opacity: control.active ? 1.0 : 0.0

        Behavior on opacity {
            NumberAnimation {
                duration: Theme.fadeDuration
                easing.type: Easing.Bezier
                easing.bezierCurve: Theme.easeStandard
            }
        }
        Behavior on color { ColorAnimation { duration: Theme.durationFast } }
    }

    background: Rectangle {
        color: "transparent"
    }
}
