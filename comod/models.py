from comod import Model

sir = Model("SIR",
            ["beta", "gamma"],
            [
                ("S", "I", "beta I / N"),
                ("I", "R", "gamma")
            ])

sis = Model("SI",
            ["beta", "gamma"],
            [
                ("S", "I", "beta I / N"),
                ("I", "S", "gamma")
            ])

seir = Model("SEIR",
             ["beta", "gamma", "delta"],
             [
                 ("S", "E", "beta I / N"),
                 ("E", "I", "gamma"),
                 ("I", "R", "delta")
             ])

sirs = Model("SIR",
             ["beta", "gamma", "epsilon"],
             [
                 ("S", "I", "beta I / N"),
                 ("I", "R", "gamma"),
                 ("R", "S", "epsilon")
             ])
