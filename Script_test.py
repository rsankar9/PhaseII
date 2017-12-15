import numpy as np
import Test_BG as model_test_BG
import Test_HL as model_test_HL
import Test_multiple_syllables as model_test_mult_syll


test = 3								# 1 -> BG 	2 -> HL 	3 -> Multiple Syllables [only BG]

if test == 1:

    f = 0
    for run in range(5):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_BG_' + str(f)
        model_test_BG.phaseIIModel(resFile, rSeed)



elif test == 2:

    f = 0
    for run in range(5):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_HL_' + str(f)
        model_test_HL.phaseIIModel(resFile, rSeed)



elif test == 3:

    f = 0
    for run in range(5):
        f = f + 1
        rSeed = np.random.randint(0, 1e7)

        resFile = 'Results/Test_multiple_syllables_' + str(f)
        model_test_mult_syll.phaseIIModel(resFile, rSeed)