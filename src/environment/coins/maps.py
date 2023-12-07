COINS_MAPS = {
    'default': {
        'map': """
        WWWWWWWWWWWWWWWWW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCC_CW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WC_CCCCCCCCCCCCCW
        WCCCCCCCCCCCCCCCW
        WWWWWWWWWWWWWWWWW
                """,
        'description': "copy of the default arena map provided by DeepMind's Melting Pot Environment. Note that the"
                       "coins environment map actually has randomly chosen dimensions, which this default shape does"
                       "not capture.",
    },

    '13x13_minified': {
        'map': """
            WWWWWWWWWWWWW
            WCCCCCCCCCCCW
            WCCCCCCCCC_CW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WCCCCCCCCCCCW
            WC_CCCCCCCCCW
            WCCCCCCCCCCCW
            WWWWWWWWWWWWW

            """,
        'description': "Smaller map to force more interaction between agents",
    }
}
