import numpy as np
import json

# Layer parameters
HVC_n = 21
RA_n = 12
DM_n = 3
cluster_size = 4                                        # no. of RA units connected to 1 DM unit

R_o = 0.0                                               # Expected reward is initially 0
sw = 50.0                                               # Window size for sliding average

ntrials = 1000

# Learning parameters
eta = 0.05                                              # Learning rate
r_std_dev = 3.0                                         # Sigma for error - reward gaussian

rSeed = np.random.randint(0,1e7)
np.random.seed(rSeed)

# Weight Bounds
w_hr_lb = 0.0
w_hr_ub = 1.0
w_rd_lb = 0.0
w_rd_ub = 1.0
mo_lb = 2.0                                             # Desired motor output lower bound
mo_ub = 6.0                                             # Desired motor output upper bound

# Neuron noise in RA transfer function
neuron_noise_mean = 0.0
neuron_noise_sd = 3.0
neuron_beta = 0.20                                      # Threshold


HVC_c = np.zeros(HVC_n, int)
RA_s = np.zeros(RA_n, float)
DM_m = np.zeros(DM_n, float)
W_hvc_ra = np.zeros((HVC_n, RA_n), float)
W_ra_dm = np.zeros((RA_n, DM_n), float)
R = np.zeros(ntrials+1, float)

euler_stuff = []                                        # debug - ignore


def calc_neuronAct_RA():
    for j in range(RA_n):
        RA_s[j] += np.random.normal(neuron_noise_mean, neuron_noise_sd) - neuron_beta   # adding noise over a threshold
        RA_s[j] = max(0, RA_s[j])

def calc_transfer_DM():
    for k in range(DM_n):
        DM_m[k] = np.dot(RA_s[k * cluster_size : (k + 1) * cluster_size], W_ra_dm[k * cluster_size:(k + 1) * cluster_size][:, k])
        # to take into account only 4 (= cluster_size) RA units for 1 DM unit [1*4 dot 4*1 = 1*1]

        # DM_m[k] += np.random.normal(neuron_noise_mean, neuron_noise_sd)
        DM_m[k] -= neuron_beta
        DM_m[k] = max(0, DM_m[k])

def calc_reward(DM_m, DM_optimal):
    E_d = Euler_distance(DM_m, DM_optimal)
    return np.exp(-(pow(E_d, 2) / (pow(r_std_dev, 2))))

def calc_sliding_average(R_o, R, nt):
    if nt<sw:   R_o = (R_o * nt + R[nt])/(nt+1)
    else:       R_o = (R_o*sw + R[nt] - R[nt-sw])/sw
    return R_o

def Euler_distance(dm_m, dm_optimal):
    sum = 0.0
    for k in range(DM_n):
        sum += pow(dm_m[k]-dm_optimal[k],2)
    return np.sqrt(sum)

def calc_learning_rule(R_t, R_o):
    delta_w = eta * (R_t-R_o) * RA_s[j] * HVC_c[i]
    return delta_w

# HVC input
SyllableEncoding = {
    "A" : np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    "B" : np.array([0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    "C" : np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]),
    "D" : np.array([0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]),
    "E" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]),
    "F" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0]),
    "G" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]),
    }

# randomly picking desired motor output
MotorOutput = {
    "A" : np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "B": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "C": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "D": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "E": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "F": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    "G": np.array([np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub), np.random.uniform(mo_lb, mo_ub)]),
    }

# Initialising HVC-RA weights
for i, j in np.ndindex(HVC_n, RA_n):
    W_hvc_ra[i,j] = np.random.uniform(w_hr_lb, w_hr_ub)

# Initialising RA-DM weights
for i, j in np.ndindex(RA_n, DM_n):
    W_ra_dm[i,j] = np.random.uniform(w_rd_lb, w_rd_ub)      # fixed i.e. does not change in training


seq1 = ["A"]                                                # seq1 = ["A", "B", "C"]
training_sequences = [seq1]                                 # training_sequences = [seq1, seq2]
results = []

for nt in range(ntrials):
    for seq in training_sequences:
        for t in range(len(seq)):
            syll = seq[t]
            HVC_c = SyllableEncoding[syll]

            RA_s = np.dot(HVC_c, W_hvc_ra)
            calc_neuronAct_RA()

            calc_transfer_DM()

            # if nt == 0:
            #     MotorOutput[syll][...] = (DM_m)
            #     R[0] = 1.0
            # To check if the output doesn't diverge, I set the output generated in the first trial as the desired output

            # Reward calculation
            R[nt+1] = calc_reward(DM_m, MotorOutput[syll])
            R_o = calc_sliding_average(R_o, R, nt)

            # Learning
            for i, j in np.ndindex(HVC_n, RA_n):
                delta_w = calc_learning_rule(R[nt+1], R_o)
                W_hvc_ra[i,j] += delta_w
                W_hvc_ra[i,j] = max(w_hr_lb, W_hvc_ra[i,j])
                W_hvc_ra[i,j] = min(w_hr_ub, W_hvc_ra[i,j])

            E_d = Euler_distance(DM_m, MotorOutput[syll])
            print "desired mo:", MotorOutput[syll]
            print "produced mo:", DM_m
            print "Euler:", E_d
            euler_stuff.append(E_d)

            # Json result
            if (nt+1)%50==0:
                desired_result = str(MotorOutput[syll])
                produced_result = str(DM_m)
                Euler_result = str(E_d)
                current_result = [desired_result, produced_result, Euler_result]
                results.append(current_result)

# --- Write to JSON file --- #

layer_parameters = {
    "HVC population": HVC_n,
    "RA population": RA_n,
    "DM population": DM_n
}

learning_parameters = {
    "RA cluster per DM": cluster_size,
    "No. of trials": ntrials,
    "Sliding average window size": sw,
    "Learning rate": eta,
    "Reward std deviation": r_std_dev,
    "Neuron noise mean": neuron_noise_mean,
    "Neuron noise sigma": neuron_noise_sd,
    "Neuron threshold": neuron_beta
}

bound_parameters = {
    "HVC-RA weight lower": w_hr_lb,
    "HVC-RA weight upper": w_hr_ub,
    "RA-DM weight lower": w_rd_lb,
    "RA-DM weight upper": w_rd_ub,
    "Motor output lower": mo_lb,
    "Motor output upper": mo_ub
}

sequences_parameters = {
    "Training Sequences": training_sequences
}

input_parameters = {
    "Layer Parameters": layer_parameters,
    "Learning Parameters": learning_parameters,
    "Sequences": sequences_parameters
}


Data = {
    "GitHash": "",
    "Random seed": rSeed,
    "Input": input_parameters,
    "Results": results,
    # "Euler": euler_stuff
}

resFile = "Run1.json"
with open(resFile, 'w') as outfile:
    json.dump(Data, outfile, sort_keys=True, indent=4, separators=(',', ':\t'))

