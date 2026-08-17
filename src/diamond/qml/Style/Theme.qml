pragma Singleton
import QtQuick

// Every colour, size and duration in the UI comes from here.
// Components must not introduce their own colour literals.
QtObject {
    // -- surfaces ---------------------------------------------------------
    readonly property color background:   "#EEEEE9"
    readonly property color surface:      "#FFFFFF"
    readonly property color surfaceAlt:   "#F6F6F2"
    readonly property color border:       "#DBDBD4"
    readonly property color borderStrong: "#BFBFB7"

    // -- board ------------------------------------------------------------
    readonly property color boardBackground: "#FFFFFF"
    readonly property color lattice:         "#1B1B1B"
    readonly property color hole:            "#121212"
    readonly property color campNeutral:     "#D5D7DA"

    // -- text -------------------------------------------------------------
    readonly property color text:       "#15171A"
    readonly property color textMuted:  "#6B7077"
    readonly property color textFaint:  "#9AA0A6"
    readonly property color textOnDark: "#FFFFFF"

    // -- interaction accents (restrained, no glow) ------------------------
    readonly property color selection:  "#1F1F1F"
    readonly property color legalStep:  "#2E7D57"
    readonly property color legalJump:  "#B0731A"
    readonly property color pathLine:   "#2F6FA8"
    readonly property color proposal:   "#2F6FA8"
    readonly property color lastMove:   "#9AA0A6"
    readonly property color danger:     "#B3261E"
    readonly property color thinking:   "#8A6D1F"

    // -- geometry ---------------------------------------------------------
    readonly property real  boardPadding:   28
    readonly property real  holeRatio:      0.17   // of one lattice edge
    readonly property real  pieceRatio:     0.36
    readonly property real  latticeWidth:   1.1
    // Camp triangles are tinted rather than fully saturated so that a piece
    // standing in its own camp still reads clearly against the fill.
    readonly property real  campFillAlpha:  0.40
    readonly property real  panelWidth:     356
    readonly property real  radiusSmall:    3
    readonly property real  radiusMedium:   5
    readonly property real  spacing:        10
    readonly property real  spacingLarge:   16

    // -- typography -------------------------------------------------------
    // `appFontFamily` is set from Python after the bundled Google Sans Flex
    // files are registered with Qt, so the name here always resolves.  The
    // literal is only a safety net for tooling that loads QML standalone.
    readonly property string fontFamily: (typeof appFontFamily !== "undefined" && appFontFamily)
                                         ? appFontFamily
                                         : "Google Sans Flex"

    // Note: this Qt build's QML `font` value type has no `families` list, so
    // components bind the single `fontFamily`.  Google Sans Flex is a Latin
    // face with no arrow glyphs, but Qt substitutes those from the system font
    // database automatically, so move notation ("12 → 34") still renders.
    // Icons are drawn as shapes rather than typed, so they never depend on it.
    readonly property int    fontTiny:   11
    readonly property int    fontSmall:  12
    readonly property int    fontBody:   14
    readonly property int    fontLarge:  17
    readonly property int    fontTitle:  20
    readonly property int    fontHuge:   26

    // Google Sans Flex ships Regular/Medium/Bold; naming the weights keeps
    // components off `font.bold`, which would synthesise a fake bold.
    readonly property int    weightRegular: Font.Normal
    readonly property int    weightMedium:  Font.Medium
    readonly property int    weightBold:    Font.Bold

    // -- motion -----------------------------------------------------------
    readonly property int hopDuration:   120
    readonly property int fadeDuration:  120
}
