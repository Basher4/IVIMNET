from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import pickle, gzip
import numpy as np
import time
import torch
import os, sys

sys.path.append("./")
sys.path.append("./../")
sys.path.append("./../../")

from hyperparams import hyperparams as hp_example_1
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
import IVIMNET.deep_simplified as deep_simpl
import IVIMNET.fitting_algorithms as fit

SAMPLES     = 100
EP_DROPOUT  = [0.075, 0.125, 0.25]   # Probability of setting a weight to 0
EP_SNR      = [5, 15, 20, 30, 50]    # Train network for an SNR  and evaluate it
MODEL_NAME_TMPL = './evaluation/models/bnn_p{}_snr{}.pt'
SIGNAL_NAME_TMPL= './evaluation/signals/infer_{}SNR.pickle.gz'

PLOT_MODE = "mean"
PLOT_PATH = "./evaluation/plots/signals_err"

arg = hp_example_1()
arg = deep.checkarg(arg)
arg.train_pars.device = "cuda"
infer_signals = {}

def sample_net(net, dwi_image_long):
    params = defaultdict(list)

    start_time = time.time()
    for i in range(SAMPLES):
        print(f'Sampling - {i+1}/{SAMPLES}', end='\r')
        X, Dt, Fp, Ds, f0 = deep_simpl.predict_IVIM(dwi_image_long, arg.sim.bvalues, net, arg, signals_out=True)
        params['X'].append(X)
        params['Dt'].append(Dt)
        params['Fp'].append(Fp)
        params['Ds'].append(Ds)
        params['f0'].append(f0)

    elapsed_time = time.time() - start_time
    print('Sampling done!                 ')
    print(f'time elapsed for inference - {SAMPLES} samples: {elapsed_time:.3f}s\n')
    return params

def plot_ax(bvals, snr, col, err, fig, ax):
    for i in range(11):
        dat = err[i].reshape((100,100))
        ax[i, col].set_title(f"SNR={snr} bvalue={bvals[i]}")
        ax[i, col].set_xticks([])
        ax[i, col].set_yticks([])
        plot = ax[i, col].imshow(dat, cmap='gray', clim=(0, 0.05))
        fig.colorbar(plot, ax=ax[i, col])

def plot_signal_err(input, output, bvals, snr, col, fig, ax):
    input = deep.normalise(input, bvals, arg)
    out_mean = np.mean(output)
    out_std = np.std(output, axis=0).T

    # mean
    err = (input - out_mean).T
    plot_ax(bvals, snr, col, err, fig["mean"], ax["mean"])

    # L2
    err = np.sqrt(err * err)
    plot_ax(bvals, snr, col, err, fig["l2"], ax["l2"])

    # std
    err = out_std
    plot_ax(bvals, snr, col, err, fig["std"], ax["std"])

    # cov
    err = out_std / out_mean
    plot_ax(bvals, snr, col, err, fig["cov"], ax["cov"])
    


def evaluate(dropout, snr):
    net = torch.load(MODEL_NAME_TMPL.format(dropout, snr), map_location=arg.train_pars.device)

    fig_mean, ax_mean = plt.subplots(11, 5, figsize=(10,18), constrained_layout=True)
    fig_l2, ax_l2 = plt.subplots(11, 5, figsize=(10,18), constrained_layout=True)
    fig_std, ax_std = plt.subplots(11, 5, figsize=(10,18), constrained_layout=True)
    fig_cov, ax_cov = plt.subplots(11, 5, figsize=(10,18), constrained_layout=True)
    fig = {"l2": fig_l2, "mean": fig_mean, "std": fig_std, "cov": fig_cov}
    ax = {"l2": ax_l2, "mean": ax_mean, "std": ax_std, "cov": ax_cov}

    fig_mean.suptitle(f"Signal error (NN trained with SNR={snr} and p={dropout})", fontsize=16)
    fig_l2.suptitle(f"L2 distance (NN trained with SNR={snr} and p={dropout})", fontsize=16)
    fig_std.suptitle(f"Std. dev. of samples (NN trained with SNR={snr} and p={dropout})", fontsize=16)
    fig_cov.suptitle(f"σ/µ of samples (NN trained with SNR={snr} and p={dropout})", fontsize=16)

    for col, net_snr in enumerate(EP_SNR):
        dwi_image_long = infer_signals[net_snr][0]
        params = sample_net(net, dwi_image_long)
        X = np.reshape(params['X'], (SAMPLES, *dwi_image_long.shape))
        plot_signal_err(dwi_image_long, X, arg.sim.bvalues, net_snr, col, fig, ax)

    fig_mean.savefig(f"{PLOT_PATH}_mean/snr{snr}-p{dropout}.png")
    fig_l2.savefig(f"{PLOT_PATH}_l2/snr{snr}-p{dropout}.png")
    fig_std.savefig(f"{PLOT_PATH}_std/snr{snr}-p{dropout}.png")
    fig_cov.savefig(f"{PLOT_PATH}_cov/snr{snr}-p{dropout}.png")

def simulate_signals():
    start_time = time.time()
    for snr in EP_SNR:
        train_name = SIGNAL_NAME_TMPL.format(snr)
        if os.path.exists(train_name):
            with gzip.open(train_name, "rb") as fd:
                infer_signals[snr] = pickle.load(fd)
        else:
            print(f'L2 error of signals simulated at SNR {snr}')
            infer_signals[snr] = sim.sim_signal_predict(arg, snr)
            # TODO: How the fuck do I save a tuply of numpy arrays? np.save() doesn't work. Why tho?
            with gzip.open(train_name, "wb") as fd:
                pickle.dump(infer_signals[snr], fd)
    elapsed_time = time.time() - start_time
    print(f'\nTime to simulate {len(EP_SNR)} signal sets: {elapsed_time:.3f}s\n')

if __name__ == "__main__":
    simulate_signals()

    # Exhaustive parameter search.
    for dropout in EP_DROPOUT:
        for snr in EP_SNR:
            print(f"Evaluating model for SNR {snr} and dropout {dropout}")
            evaluate(dropout, snr)
