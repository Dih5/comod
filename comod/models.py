from comod import Model

sir = Model("SIR",
            "bg",
            [
                ("S", "I", "b I / N"),
                ("I", "R", "g")
            ])

sis = Model("SI",
            "bg",
            [
                ("S", "I", "b I / N"),
                ("I", "S", "g")
            ])

seir = Model("SEIR",
             "bgd",
             [
                 ("S", "E", "b I / N"),
                 ("E", "I", "d"),
                 ("I", "R", "g")
             ])
