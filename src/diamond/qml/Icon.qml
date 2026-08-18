import QtQuick
import Style

// A QtAwesome glyph, rendered by the Python-side image provider.
//
// `name` is a QtAwesome id such as "msc.chrome-close" or "fa6s.diamond".
// The colour is passed without its leading "#", which would otherwise
// terminate the URL.
Image {
    id: root

    property string name: ""
    property color color: Theme.text
    property int size: 16

    // Render at device resolution so the glyph stays crisp when the window is
    // on a scaled display.
    readonly property real _dpr: Screen.devicePixelRatio

    width: size
    height: size
    sourceSize.width: Math.round(size * _dpr)
    sourceSize.height: Math.round(size * _dpr)
    smooth: true
    fillMode: Image.PreserveAspectFit

    source: name === ""
            ? ""
            : "image://qta/" + name + "/"
              + Qt.rgba(color.r, color.g, color.b, 1).toString().substring(1)
}
