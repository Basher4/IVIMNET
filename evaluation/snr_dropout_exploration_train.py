from collections import defaultdict
from datetime import datetime
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
SIGNAL_NAME_TMPL= './evaluation/signals/train_{}SNR.pickle.gz'

arg = hp_example_1()
arg = deep.checkarg(arg)
arg.train_pars.device = "cuda"

train_signals = {}

def evaluate(dropout, snr):
    net_params = deep_simpl.net_params()
    net_params.dropout = dropout
    IVIM_signal_noisy, D, f, Dp = train_signals[snr]

    # Train the model unless we have one trained.
    model_name = MODEL_NAME_TMPL.format(dropout, snr)
    if os.path.exists(model_name):
        return

    net, epochs, validation_loss = deep_simpl.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg, net_params=net_params, stats_out=True)
    torch.save(net, model_name)
    with open(model_name + ".log", "w") as fd:
        fd.write(f"Elapsed epochs: {epochs}\n\rValidation loss: {validation_loss}\n\r")
    
    
def simulate_signals():
    start_time = time.time()
    for snr in EP_SNR:
        print(f'Simulating signals for SNR {snr}')
        train_name = SIGNAL_NAME_TMPL.format(snr)
        if os.path.exists(train_name):
            with gzip.open(train_name, "rb") as fd:
                train_signals[snr] = pickle.load(fd)
        else:
            train_signals[snr] = sim.sim_signal(snr, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                                Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                                fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                                Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
            # TODO: How the fuck do I save a tuply of numpy arrays? np.save() doesn't work. Why tho?
            with gzip.open(train_name, "wb") as fd:
                pickle.dump(train_signals[snr], fd)
            elapsed_time = time.time() - start_time
    print(f'\nTime to simulate {len(EP_SNR)} signal sets: {elapsed_time:.3f}s\n')

if __name__ == "__main__":
    simulate_signals()

    # Exhaustive parameter search.
    for dropout in EP_DROPOUT:
        for snr in EP_SNR:
            print(f"Training model for SNR {snr} and dropout {dropout}")
            evaluate(dropout, snr)