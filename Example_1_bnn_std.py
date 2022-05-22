#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)

This is example 1 modified to verify that my simplification before using BNNs works as expected.
"""
from collections import defaultdict
from datetime import datetime
import numpy as np
import time
import torch
import sys

from hyperparams import hyperparams as hp_example_1
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
import IVIMNET.deep_simplified as deep_simpl
import IVIMNET.fitting_algorithms as fit

# Import parameters
arg = hp_example_1()
arg = deep.checkarg(arg)
# arg.fit.do_fit = False  # Skip lsq fitting.

SAMPLES = 200
TRAIN_MODEL = True

print(arg.save_name)

# for SNR in arg.sim.SNR:
# SNRs are [15, 20, 30, 50]

# this simulates the signal
SNR = int(sys.argv[1] if len(sys.argv) > 1 else 15)
EPOCHS = 1000
IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                            Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                            fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                            Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)

start_time = time.time()
# train network
bvalues = torch.FloatTensor(arg.sim.bvalues[:]).to(arg.train_pars.device)

if TRAIN_MODEL:
    net = deep_simpl.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg, epochs=EPOCHS, bayes_samples=SAMPLES)
else:
    net = deep_simpl.Net(bvalues, deep_simpl.net_params()).to("cuda")
    net.load_state_dict(torch.load('./models/11bvs_nn_relu_net_dict_1000eps_15SNR.pt'))

elapsed_time = time.time() - start_time
print(f'\ntime elapsed for training: {elapsed_time}\n')

# simulate IVIM signal for prediction
[dwi_image_long, Dt_truth, Fp_truth, Dp_truth] = sim.sim_signal_predict(arg, SNR)

# predict
print('\nEvaluating the BNN\n')
print(f'Sampling - 0/{SAMPLES}', end='\r')
params = defaultdict(list)
start_time = time.time()
for i in range(SAMPLES):
    print(f'Sampling - {i+1}/{SAMPLES}', end='\r')
    Dt, Fp, Ds, f0 = deep_simpl.predict_IVIM(dwi_image_long, arg.sim.bvalues, net, arg)
    params['Dt'].append(Dt)
    params['Fp'].append(Fp)
    params['Ds'].append(Ds)
    params['f0'].append(f0)
elapsed_time = time.time() - start_time
print('Sampling done!                 ')
print(f'time elapsed for inference - {SAMPLES} samples: {elapsed_time:.3f}s\n')

truth = {
    'Dt': Dt_truth,
    'Fp': Fp_truth,
    'Dp': Dp_truth
}

# Save model
if TRAIN_MODEL:
    torch.save(net, f"./models/{len(arg.sim.bvalues)}bvs_bnn-{SAMPLES}ts_relu_net_{EPOCHS}epochs_{SNR}SNR.pt")

# Print stats
with open('nn_relu_stdev.txt', 'a') as fd:
    print(f'Standard deviation estimates for SNR {SNR}:')
    fd.write(f'Standard deviation estimates for SNR {SNR}:\n')

    for pp in 'Dt Fp Ds f0'.split():
        std = np.std(params[pp], axis=1)
        mean = np.mean(params[pp], axis=1)
        coef_of_var = std / mean

        print(f"\t{pp}: Stdev: (mean={np.mean(std):.6f}, med={np.median(std):.6f}, min={np.min(std):.6f}, max={np.max(std):.6f})")
        print(f"\t    CofV:  (mean={np.mean(coef_of_var):.6f}, med={np.median(coef_of_var):.6f}, min={np.min(coef_of_var):.6f}, max={np.max(coef_of_var):.6f})")
        fd.write(f"\t{pp}: Stdev: (mean={np.mean(std):.6f}, med={np.median(std):.6f}, min={np.min(std):.6f}, max={np.max(std):.6f})\n")
        fd.write(f"\t    CofV:  (mean={np.mean(coef_of_var):.6f}, med={np.median(coef_of_var):.6f}, min={np.min(coef_of_var):.6f}, max={np.max(coef_of_var):.6f})\n")

        # if pp != 'f0':
        #     err = params[pp] - truth[pp]
        #     print(f"\t    Ground truth error: (mean={np.mean(err):.6f}, med={np.median(err):.6f}")
        #     fd.write(f"\t    Ground truth error: (mean={np.mean(err):.6f}, med={np.median(err):.6f}\n")
    fd.write("\n\n")

# remove network to save memory
del net
if arg.train_pars.use_cuda:
    torch.cuda.empty_cache()

# plot values predict and truth
# paramsNN = (Dt, Fp, Ds, f0)
# print(paramsNN)

paramsNN = np.mean([params[x] for x in 'Dt Fp Ds f0'.split()], axis=1)

paramsf = fit.fit_dats(arg.sim.bvalues, dwi_image_long, arg.fit)
sim.plot_example1(params, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, SNR, prefix=f'bnn-{SAMPLES}ts_relu_')
