import numpy as np
import Model_RL22 as modelSigmoidal


f = 0

for run in range(10):
    f = f + 1
    rSeed = np.random.randint(0, 1e7)

    resFile = 'Results/Model22_Final_' + str(f)
    modelSigmoidal.phaseIIModel(resFile, rSeed)