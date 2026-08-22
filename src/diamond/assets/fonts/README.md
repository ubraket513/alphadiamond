# Bundled fonts

**Google Sans Flex** — the application's typeface, used for every piece of UI text.

| | |
|---|---|
| Source | <https://fonts.google.com/specimen/Google+Sans+Flex> |
| License | SIL Open Font License 1.1 (see `OFL.txt`) |
| Weights | Regular (400), Medium (500), Bold (700) |

Bundled rather than assumed installed: the font ships with no desktop OS, and a
missing family would silently fall back to a system default, changing every
metric in the layout. The native Qt resource setup registers these
files with Qt at startup and returns the resolved family name, so QML asks for a
family that is guaranteed to exist.

The static per-weight TTFs are what Google Fonts serves to clients that do not
advertise WOFF2 support. They are used in preference to the variable font
because Qt's variable-font axis support is inconsistent across platforms.
