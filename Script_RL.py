import numpy as np
import Model_only_RL as modelSigmoidal


f = 0

for run in range(10):
    f = f + 1
    rSeed = np.random.randint(0, 1e7)

    resFile = 'Results/' + str(f)
    modelSigmoidal.phaseIIModel(resFile, rSeed)
