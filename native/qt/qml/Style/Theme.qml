pragma Singleton
import QtQuick

// Every colour, size and duration in the UI comes from here.
// Components must not introduce their own colour literals.
//
// Palette: the Apple Human Interface Guidelines system colours, light
// appearance (https://developer.apple.com/design/human-interface-guidelines/color).
// Apple asks apps to read these through system APIs rather than hard-code them,
// because the values shift between releases -- but this is a Qt desktop app
// with no such API to read, so the documented values are pinned here and kept
// in this one file.
//
// Roles:
//   * systemBlue is the single UI accent -- selection, focus, the confirming
//     action, the proposed move path.  No player is blue, so "blue means the
//     interface" never collides with "this colour means a player".
//   * Player identity is red / yellow / green, defined once in SEAT_LAYOUTS
//     (game/state.py) and delivered to QML through the models.
//   * Greys carry structure; saturated colour is reserved for meaning.
QtObject {
    // -- Apple system colours (light appearance) --------------------------
    readonly property color systemRed:    "#FF3B30"
    readonly property color systemOrange: "#FF9500"
    readonly property color systemYellow: "#FFCC00"
    readonly property color systemGreen:  "#34C759"
    readonly property color systemTeal:   "#30B0C7"
    readonly property color systemBlue:   "#007AFF"
    readonly property color systemIndigo: "#5856D6"
    readonly property color systemGray:   "#8E8E93"
    readonly property color systemGray2:  "#AEAEB2"
    readonly property color systemGray3:  "#C7C7CC"
    readonly property color systemGray4:  "#D1D1D6"
    readonly property color systemGray5:  "#E5E5EA"
    readonly property color systemGray6:  "#F2F2F7"

    // Player colours are NOT defined here: they depend on the seat layout
    // (see SEAT_LAYOUTS in game/state.py) and reach QML through the models.

    // -- surfaces ---------------------------------------------------------
    readonly property color background:   systemGray6
    readonly property color surface:      "#FFFFFF"
    readonly property color surfaceAlt:   "#FAFAFC"
    readonly property color border:       systemGray5
    readonly property color borderStrong: systemGray3
    readonly property color shadowSoft:   "#18000000"

    // -- board ------------------------------------------------------------
    readonly property color boardBackground: "#FFFFFF"
    // The lattice is structure, not content: grey, so the coloured camps and
    // pieces are the only saturated things on the board.
    readonly property color lattice:         "#D3D3D8"
    readonly property color hole:            "#B8B8BF"
    readonly property color campNeutral:     systemGray5

    // -- text (HIG label hierarchy) ---------------------------------------
    readonly property color text:       "#000000"
    readonly property color textMuted:  "#6C6C70"   // secondaryLabel, flattened
    readonly property color textFaint:  "#A1A1A6"   // tertiaryLabel, flattened
    readonly property color textOnDark: "#FFFFFF"

    // -- interaction accents ----------------------------------------------
    readonly property color accent:     systemBlue
    readonly property color selection:  systemBlue
    // Step and jump share one treatment; the distinction stays in the engine
    // but is not worth a second colour on the board.
    readonly property color legalMove:  systemBlue
    readonly property color pathLine:   systemBlue
    readonly property color proposal:   systemBlue
    readonly property color lastMove:   systemGray2
    readonly property color danger:     systemRed
    readonly property color thinking:   systemOrange
    readonly property color success:    systemGreen

    // -- geometry ---------------------------------------------------------
    readonly property real  boardPadding:   28
    // A piece exactly fills a hole, so one radius governs both: the hole is a
    // hollow socket and an occupied one is that socket filled with the owner's
    // colour. In lattice units, against a spacing of 1.0.
    readonly property real  socketRatio:    0.32
    readonly property real  socketStroke:   0.035  // ring weight, lattice units
    readonly property real  latticeWidth:   1.0

    // Pieces carry their state in opacity alone -- selected reads solid, every
    // other piece sits back. No ring, no second marker.
    readonly property real  pieceOpacity:         0.5
    readonly property real  pieceOpacitySelected: 1.0

    // A legal destination is the empty socket filled with translucent accent:
    // a ghost of the piece that would land there.
    readonly property real  ghostAlpha:     0.32

    // Camp regions: a wash, not a fill.  Low alpha keeps a piece standing in
    // its own camp legible against it; the layer sits behind everything else.
    readonly property real  campFillAlpha:  0.16

    // Custom window chrome (the window is frameless; see TitleBar.qml).
    readonly property real  titleBarHeight: 44
    readonly property real  resizeMargin:   6

    readonly property real  panelWidth:     356
    readonly property real  radiusSmall:    6
    readonly property real  radiusMedium:   10
    readonly property real  radiusLarge:    14
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
    //
    // Two curves, used everywhere, so movement across the app feels like one
    // system rather than a set of unrelated tweens:
    //
    //   standard    - symmetric ease for things that move between two states
    //   emphasized  - leaves fast and settles slowly, for things entering or
    //                 taking over the screen
    //
    // Both are cubic beziers rather than a named easing type, because Qt's
    // built-in curves stop short of the long, soft tail that reads as
    // "considered" instead of "animated".
    readonly property var easeStandard:   [0.4, 0.0, 0.2, 1.0, 1.0, 1.0]
    readonly property var easeEmphasized: [0.2, 0.0, 0.0, 1.0, 1.0, 1.0]
    // Overshoots slightly, for the one moment that should feel physical.
    readonly property var easeSpring:     [0.34, 1.4, 0.64, 1.0, 1.0, 1.0]

    readonly property int durationFast:   140
    readonly property int durationBase:   220
    readonly property int durationSlow:   300

    // A hop must finish inside the native controller's 140 ms landing timer;
    // overrun and the piece lags the tick.
    readonly property int hopDuration:   130
    readonly property int fadeDuration:  160
    readonly property int panelDuration: 260
}
