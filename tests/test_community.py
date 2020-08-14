import comod


def test_community():
    """Test the generation of community models"""
    sir = comod.Model("SIR",  # States
                      "bg",  # Coefficients
                      [  # Rules in the form (origin, destination, coefficient)
                          ("S", "I", "b I / N"),
                          ("I", "R", "g")
                      ])
    sir2 = comod.CommunityModel(sir, 2)
    assert sir2.parameters == ['b_1', 'g_1', 'b_2', 'g_2', 'm_1_2', 'm_2_1']
    assert sir2.states == ['S_1', 'I_1', 'R_1', 'S_2', 'I_2', 'R_2']
    assert sir2.rules == [('S_1', 'I_1', 'b_1 I_1 / N_1'),
                          ('I_1', 'R_1', 'g_1'),
                          ('S_2', 'I_2', 'b_2 I_2 / N_2'),
                          ('I_2', 'R_2', 'g_2'),
                          ('S_1', 'S_2', 'm_1_2 / N_1'),
                          ('I_1', 'I_2', 'm_1_2 / N_1'),
                          ('R_1', 'R_2', 'm_1_2 / N_1'),
                          ('S_2', 'S_1', 'm_2_1 / N_2'),
                          ('I_2', 'I_1', 'm_2_1 / N_2'),
                          ('R_2', 'R_1', 'm_2_1 / N_2')]
