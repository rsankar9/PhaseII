# Learning Syllable A only with target 17

import numpy as np
import matplotlib.pyplot as plt
import json

def sigmoid(x, m=5):
    return 1 / (1 + np.exp(-1*(x-0.5)*m))

def phaseIIModel(arg_resFile, arg_rSeed):
    # Parameters
    # ---------- #
    rSeed = np.random.randint(0, 1e7)       # Random seed parameters
    rSeed = arg_rSeed
    np.random.seed(rSeed)

    HVC_size = 3                            # Layer parameters
    BG_size = 4
    RA_size = 4
    MC_size = 1
    cluster_size = RA_size // MC_size
    n_clusters = MC_size

    BG_noise_mean = 0.00                    # Activation function parameters
    BG_noise_std = 0.03
    BG_threshold = 0.0005

    RA_noise_mean = 0.00
    RA_noise_std = 0.001
    RA_threshold = 0.0005

    MC_noise_mean = 0.00
    MC_noise_std = 0.001
    MC_threshold = 0.0005

    reward_window = 25                      # Reward calculation parameters
    reward_sigma = 0.03

    n_trials = reward_window + 20000
    n_lesioned_trials = n_trials // 100     # to check Hebbian learning result without BG input
    n_cut_noise = n_lesioned_trials * 5     # to check RL result without noise
    BG_noise_reduction = 0
    BG_influence = 1
    eta = 0.08                              # Learning parameters
    Hebbian_learning = 1
    Hebbian_rule = 1                        # 1 -> Hebbian, 2 -> iBCM, 3 -> Oja
    pPos = 0.001
    pDec = 0.0001
    tau = 1500                              # only used in iBCM

    soft_bound = 1                          # Weight parameters
    Wmin_ini, Wmax_ini = 0, 1
    Wmin, Wmax = 0, 1
    Wmin_Heb, Wmax_Heb = 0, 1
    Wepsilon = 0.05

    min_possible_output = 0                 # To calculate normalised error
    max_possible_output = 1
    output_range = 1

    n_bits = 3                              # no. of bits active in each syllable encoding
    n_samples = HVC_size // n_bits          # maximum syllables possible without overlap in encoding

    resFile = arg_resFile

    BG_max = RA_max = MC_max = 1
    RA_sig_slope = MC_sig_slope = BG_sig_slope = 5

    print("# ---INITIALISATIONS--- #")

    # Model build
    # ----------- #
    HVC = np.zeros(HVC_size)
    RA = np.zeros(RA_size)
    MC = np.zeros(MC_size)
    BG = np.zeros(BG_size)

    W_HVC_RA = np.zeros((HVC_size, RA_size), float) + Wepsilon
    W_RA_MC = np.random.uniform(Wmin_ini + Wepsilon, Wmax_ini - Wepsilon, (RA_size, MC_size))
    W_HVC_BG = np.random.uniform(Wmin_ini + Wepsilon, Wmax_ini - Wepsilon, (HVC_size, BG_size))
    W_BG_RA = np.random.uniform(Wmin_ini + Wepsilon, Wmax_ini - Wepsilon, (BG_size, RA_size))

    # Segregated pathways between RA and MC
    for i in range(n_clusters):
        W_RA_MC[i*cluster_size : (i+1)*cluster_size] *= np.diag(np.ones(n_clusters, int))[i]

    # Segregated pathways between BG and RA
    for i in range(n_clusters):
        segPath = np.diag(np.ones(n_clusters, int))[i]
        W_BG_RA[i*cluster_size : (i+1)*cluster_size] *= [j for j in segPath for r in range(cluster_size)]

    # Syllable encoding and outputs
    # ---------- #

    HVC = np.ones(HVC_size)
    W_HVC_BG_temp = np.zeros((HVC_size, BG_size)) + Wmin

    BG[...] = np.dot(HVC, W_HVC_BG_temp) / HVC_size - BG_threshold
    BG = sigmoid(BG, BG_sig_slope)

    RA[...] = np.dot(BG, W_BG_RA) / cluster_size - RA_threshold
    RA = sigmoid(RA, RA_sig_slope)

    MC[...] = np.dot(RA, W_RA_MC) / cluster_size - MC_threshold
    MC = sigmoid(MC, MC_sig_slope)

    min_possible_output = min(MC)

    HVC = np.ones(HVC_size)
    W_HVC_BG_temp = np.zeros((HVC_size, BG_size)) + Wmax

    BG[...] = np.dot(HVC, W_HVC_BG_temp) / HVC_size - BG_threshold
    BG = sigmoid(BG, BG_sig_slope)

    RA[...] = np.dot(BG, W_BG_RA) / cluster_size  - RA_threshold
    RA = sigmoid(RA, RA_sig_slope)

    MC[...] = np.dot(RA, W_RA_MC) / cluster_size - MC_threshold
    MC = sigmoid(MC, MC_sig_slope)

    max_possible_output = max(MC)

    output_range = max_possible_output - min_possible_output

    print("Output range:", output_range, min_possible_output, max_possible_output)

    syllable_encoding = {}
    syllables = ["A", "B", "C", "D", "E", "F", "G"]
    syllables = syllables[:n_samples]                       # ensure n_samples >= len(syllables)

    for i in range(n_samples):
        inputs = np.zeros(HVC_size)
        inputs[i * n_bits:(i + 1) * n_bits] = 1
        syllable_encoding[syllables[i]] = inputs

    outputs = np.random.uniform(min_possible_output+0.05, max_possible_output-0.05, (n_samples, MC_size))
    syllable_outputs = {}
    for i in range(n_samples):
        syllable_outputs[syllables[i]] = outputs[i]

    print('Target', syllable_outputs)

    # Simulation
    # ---------- #
    R = np.zeros(n_trials)                              # keeps track of reward
    E = np.zeros(n_trials)                              # keeps track of error
    E_norm = np.zeros(n_trials)                         # keeps track of normalised error
    W = np.zeros((n_trials, W_HVC_BG.size))             # to plot HVC-BG weights
    W_HR = np.zeros((n_trials, W_HVC_RA.size))          # to plot HVC-RA weights
    T = np.zeros(RA_size)

    syll = "A"

    for nt in range(n_trials):
        # if nt == n_trials//2:            Hebbian_learning = 1       # To have Hebbian learning influence only for half the trails

        HVC[...] = syllable_encoding[syll]

        # Compute BG 6
        BG[...] = np.dot(HVC, W_HVC_BG) / HVC_size
        BG += np.random.normal(BG_noise_mean, BG_noise_std, BG_size) - BG_threshold
        BG = sigmoid(BG, BG_sig_slope)

        # Compute RA activity with BG
        RA = np.random.normal(RA_noise_mean, RA_noise_std, RA_size) - RA_threshold
        total_RA_inputs = 0.0
        if Hebbian_learning == 1:
            RA[...] += np.dot(HVC, W_HVC_RA)
            total_RA_inputs += HVC_size
        if BG_influence == 1:
            RA[...] += np.dot(BG, W_BG_RA)
            total_RA_inputs += cluster_size
        if total_RA_inputs == 0:
            total_RA_inputs = 1.0
        RA[...] = RA[...] / total_RA_inputs
        RA = sigmoid(RA, RA_sig_slope)

        # Compute MC activity
        MC[...] = np.dot(RA, W_RA_MC) / cluster_size
        MC += np.random.normal(MC_noise_mean, MC_noise_std, MC_size) - MC_threshold
        MC = sigmoid(MC, MC_sig_slope)

        # if nt==0:         syllable_outputs[syll][...] = MC         # to check divergence

        # Compute error and reward
        error = np.sqrt(((MC - syllable_outputs[syll]) ** 2).sum())/MC_size
        norm_error = error / output_range
        E[nt], E_norm[nt] = error, norm_error
        R[nt] = np.exp(-error ** 2 / reward_sigma ** 2)

        if nt == 0: print("First trial", "MC:", MC, "error:", E[0], "Norm error:", E_norm[0])

        # If we haven't enough trials to compute the mean reward, we skip the trial
        if nt < reward_window:  continue

        # Computing mean reward
        i0, i1 = nt - reward_window, nt
        R_ = R[i0:i1].sum() / reward_window

        T += (-T + RA ** 2) / tau                           # For iBCM learning rule

        dW1 = eta * (R[nt] - R_) * HVC.reshape(HVC_size, 1) * BG * BG_influence
        dW2 = 0
        if Hebbian_rule == 1:   dW2 = pPos * HVC.reshape(HVC_size, 1) * RA * Hebbian_learning
        elif Hebbian_rule == 2:   dW2 = pPos * (R[nt]-T) * (HVC.reshape(len(HVC),1) * (RA * (RA - T)).reshape(1,len(RA))) * Hebbian_learning
        elif Hebbian_rule == 3:   dW2 = pPos * ((HVC.reshape(HVC_size, 1) * RA) - (RA * RA * W_HVC_RA)) * Hebbian_learning                                                 # Oja learning rule                                                                  RA_size) * Hebbian_learning
        dW3 = pDec * (1 - HVC.reshape(HVC_size, 1)) * RA * Hebbian_learning

        if soft_bound == 1:
            W_HVC_BG += dW1 * (Wmax - W_HVC_BG) * (W_HVC_BG - Wmin)
            W_HVC_RA += dW2 * (Wmax_Heb - W_HVC_RA) * (W_HVC_RA - Wmin_Heb)
            W_HVC_RA -= dW3 * (Wmax_Heb - W_HVC_RA) * (W_HVC_RA - Wmin_Heb)
        else:
            W_HVC_BG = np.minimum(Wmax, np.maximum(Wmin, W_HVC_BG + dW1))
            W_HVC_RA = np.minimum(Wmax, np.maximum(Wmin, W_HVC_RA + dW2))
            W_HVC_RA = np.minimum(Wmax, np.maximum(Wmin, W_HVC_RA - dW3))

        if BG_noise_reduction == 1:     BG_noise_std *= 0.9999          # Noise reduction

        if Hebbian_learning == 1 and nt == n_trials - n_lesioned_trials:    # To check Hebbian learning
            BG_influence = 0
            avg_error_before_cut = E[i0:i1].sum() / reward_window
            avg_norm_error_before_cut = E_norm[i0:i1].sum() / reward_window
            print("Before cut:", "MC:", MC, "mean error:", avg_error_before_cut, "Norm error:", avg_norm_error_before_cut)

        if Hebbian_learning == 0 and nt == n_trials - n_cut_noise:          # To check with zero noise
            BG_noise_std = 0.0
            RA_noise_std = 0.0
            MC_noise_std = 0.0
            eta = 0
            avg_error_before_cut = E[i0:i1].sum() / reward_window
            avg_norm_error_before_cut = E_norm[i0:i1].sum() / reward_window
            print("Before cut:", "MC:", MC, "mean error:", avg_error_before_cut, "Norm error:", avg_norm_error_before_cut)


        W[nt] = W_HVC_BG.ravel()                                # for plotting purposes
        W_HR[nt] = W_HVC_RA.ravel()


    print("Final MC", MC, "error:", E[n_trials-1], "Norm error:", E_norm[n_trials-1])

    # Display results
    # ---------------
    def sliding_window(Z, window):
        shape = Z.shape[:-1] + (Z.shape[-1] - window + 1, window)
        strides = Z.strides + (Z.strides[-1],)
        return np.lib.stride_tricks.as_strided(Z, shape=shape, strides=strides)


    plt.figure(figsize=(12, 8))
    plt.rc("ytick", labelsize="small")
    plt.rc("xtick", labelsize="small")

    ax = plt.subplot(3, 1, 1)
    ax.set_title("Testing HL with normalised sigmoid")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot all trial rewards
    T = np.arange(len(R))
    ax.plot(T, R, marker="o", markersize=1.5, linewidth=0, alpha=.25,
            color="none", markeredgecolor="none", markerfacecolor="black")

    # Plot a sliding average over n=50 trials
    # n = 4 * reward_window
    n = reward_window
    Sm = np.std(sliding_window(R, n), axis=-1)
    Rm = np.mean(sliding_window(R, n), axis=-1)
    T = n - 1 + np.arange(len(Rm))
    ax.plot(T, Rm, linewidth=0.75, color="black")
    ax.text(T[-1], Rm[-1], " %.2f" % Rm[-1], color="black",
            ha="left", va="center", fontsize="small")

    ax.fill_between(T, Rm - Sm, Rm + Sm, color="black", alpha=0.1)
    # ax.set_xlabel("Trial #")
    ax.set_ylabel("Averaged reward (n=%d)" % n)

    ax.axvline(n, linestyle="--", linewidth=0.5, color="black")

    ax = ax.twinx()
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Averad error (n=%d)' % n, color='r')
    Em = np.mean(sliding_window(E, n), axis=-1)
    T = n - 1 + np.arange(len(Em))
    ax.plot(T, Em, linewidth=0.75, color="red")
    ax.text(T[-1], Em[-1], " %.2f" % Em[-1], color="red",
            ha="left", va="center", fontsize="small")

    ax.fill_between(T, Em - Sm, Em + Sm, color="red", alpha=0.1)
    ax.tick_params('y', colors="red")
    ax.spines['right'].set_color("red")

    ax = plt.subplot(3, 1, 2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    T = np.arange(n_trials)
    for i in range(W.shape[1]):
        ax.plot(T, W[:, i], color='black', alpha=.5, linewidth=0.5)
    ax.set_xlabel("Trial #")
    ax.set_ylabel("HVC - BG weights")

    ax = plt.subplot(3, 1, 3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    T = np.arange(n_trials)
    for i in range(W_HR.shape[1]):
        ax.plot(T, W_HR[:, i], color='black', alpha=.5, linewidth=0.5)
    ax.set_xlabel("Trials #")
    ax.set_ylabel("HVC - RA weights")

    plt.tight_layout()
    plt.savefig(resFile + ".pdf")
    # plt.show()


    # --- Write to JSON file --- #

    layer_parameters = {
        "HVC population": HVC_size,
        "BG population": BG_size,
        "RA population": RA_size,
        "MC population": MC_size,
        "Cluster size": cluster_size
    }

    activation_function_parameters = {
        "BG noise mean": BG_noise_mean,
        "BG noise sigma": BG_noise_std,
        "BG threshold": BG_threshold,
        "RA noise mean": RA_noise_mean,
        "RA noise sigma": RA_noise_std,
        "RA threshold": RA_threshold,
        "MC noise mean": MC_noise_mean,
        "MC noise sigma": MC_noise_std,
        "MC threshold": MC_threshold
    }

    learning_parameters = {
        "No. of trials": n_trials,
        "No. of lesioned trials": n_lesioned_trials,
        "BG noise reduction (ON/OFF)": BG_noise_reduction,
        "Learning rate": eta,
        "Hebbian learning (ON/OFF)": Hebbian_learning,
        "Hebbian learning rule (Hebb/BCM/Oja)": Hebbian_rule,
        "Hebbian learning rate": pPos,
        "Hebbian decay rate": pDec,
        "Hebbian iBCM tau": tau,
        "Reward std deviation": reward_sigma,
        "Reward average window": reward_window
    }

    bound_parameters = {
        "Soft Bound (ON/OFF)": soft_bound,
        "Weight initialisation lower": Wmin_ini,
        "Weight initialisation upper": Wmax_ini,
        "Weight lower": Wmin,
        "Weight upper": Wmax,
        "Hebbian weight lower": Wmin,
        "Hebbian weight upper": Wmax,
        "Weight margin initialisation": Wepsilon,
        "Output lower": min_possible_output,
        "Output upper": max_possible_output
    }

    sequences_parameters = {
        # "Training Sequences": training_sequences,
        "No. of bits on per encoding": n_bits,
        "No. of max syllables": n_samples,
        "Target output": str(syllable_outputs["A"])
    }

    input_parameters = {
        "Layer Parameters": layer_parameters,
        "Activation Function Parameters": activation_function_parameters,
        "Learning Parameters": learning_parameters,
        "Bounds": bound_parameters,
        "Sequences": sequences_parameters
    }

    results = {
        "Output range": str([output_range, min_possible_output, max_possible_output]),
        "First trial": str([E[0], E_norm[0]]),
        "Before cut mean error": str([avg_error_before_cut, avg_norm_error_before_cut]),
        "Final": str([E[n_trials - 1], E_norm[n_trials - 1]]),
        "Error": str(E.tolist()),
        "NormE": str(E_norm.tolist()),
        "Reward": str(R.tolist())
    }

    Data = {
        "Purpose": "Learning Syllable A only",
        # "GitHash": "",
        "Random seed": rSeed,
        "Input": input_parameters,
        "Results": results
    }

    print("Writing to json:", resFile)
    with open(resFile + ".json", 'w') as outfile:
        json.dump(Data, outfile, sort_keys=False, indent=4, separators=(',', ':\t'))

    return