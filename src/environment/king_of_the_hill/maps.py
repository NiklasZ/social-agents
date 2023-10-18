# This file collects playable maps for the KoTH environment.
# NOTE: try not to modify existing entries and instead add new ones
# ==Key==
# P: Possible spawn location for players on team 1
# Q: Possible spawn location for players on team 2
# W: Impassible wall
# D: Destructible wall (90% of the time) or destroyed wall (10% of the time)
# H: Destructible wall (75% of the time) or destroyed wall (25% of the time)
# G: Capturable hill tiles
# ,: Empty ground tile
# I: Indicator tile (shows who currently has the hill).

# Dimensions should be height, width

KING_OF_THE_HILL_MAPS = {
    'default': {
        'map': """
            IIIIIIIIIIIIIIIIIIIIIII
            IWWWWWWWWWWWWWWWWWWWWWI
            IWPPP,PPPP,P,PPPP,PPPWI
            IWPPP,,PP,,,,,PP,,PPPWI
            IWPPP,,,,,,,,,,,,,PPPWI
            IW,,,WW,,,,,,,,,WW,,,WI
            IW,,,WWDWWWDWWW,WW,,,WI
            IW,,,,,,,,,,,,,,D,,,,WI
            IW,,,,W,GGGGGGG,W,,,,WI
            IWHWWHW,GGGGGGG,WHWWHWI
            IWHWWHW,GGGGGGG,WHWWHWI
            IW,,,,D,GGGIGGG,D,,,,WI
            IWHWWHW,GGGGGGG,WHWWHWI
            IWHWWHW,GGGGGGG,WHWWHWI
            IW,,,,W,GGGGGGG,W,,,,WI
            IW,,,,D,,,,,,,,,,,,,,WI
            IW,,,WW,WWWDWWWDWW,,,WI
            IWQ,,WW,,,,,,,,,WW,,QWI
            IWQQQ,,,,,,,,,,,,,QQQWI
            IWQQQ,,QQ,,,,,QQ,,QQQWI
            IWQQQ,QQQQ,Q,QQQQ,QQQWI
            IWWWWWWWWWWWWWWWWWWWWWI
            IIIIIIIIIIIIIIIIIIIIIII
            """,

        'description': "copy of the default 23x23 map provided by DeepMind's Melting Pot Environment",
    },
    '13x13_minified': {
        'map': """
            IIIIIIIIIIIII
            IWWWWWWWWWWWI
            IWPPPP,PPPPWI
            IWWD,,D,,DWWI
            IW,,,DGD,,,WI
            IW,,,GGG,,,WI
            IW,GGGIGGG,WI
            IW,,,GGG,,,WI
            IW,,,DGD,,,WI
            IWWD,,D,,DWWI
            IWQQQQ,QQQQWI
            IWWWWWWWWWWWI
            IIIIIIIIIIIII
               """,

        'description': "Smaller map with minor elements of randomisation",
    },
    '13x13_minified_v2': {
        'map': """
            IIIIIIIIIIIII
            IWWWWWWWWWWWI
            IWPPPP,PPPPWI
            IWWD,,D,,DWWI
            IW,,,DGD,,,WI
            IW,,,GGG,,,WI
            IW,,GGIGG,,WI
            IW,,,GGG,,,WI
            IW,,,DGD,,,WI
            IWWD,,D,,DWWI
            IWQQQQ,QQQQWI
            IWWWWWWWWWWWI
            IIIIIIIIIIIII
           """,

        'description': "Fixes small visual issue with hill vs. original '13x13_minified'",
    }
}
