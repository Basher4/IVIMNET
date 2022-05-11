from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import sys

from hyperparams import hyperparams as hp_example_1
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
import IVIMNET.deep_simplified as deep_simpl
import IVIMNET.fitting_algorithms as fit


def plot_example4(params, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, path):
    paramsNN = np.mean([params[x] for x in 'Dt Fp Ds f0'.split()], axis=1)
    paramsSTD = np.std([params[x] for x in 'Dt Fp Ds f0'.split()], axis=1)

    # initialise figure
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    if arg.fit.do_fit:
        fig, ax = plt.subplots(5, 3, figsize=(20, 20))
    else:
        fig, ax = plt.subplots(4, 3, figsize=(20, 20))


    # fill Figure with values
    Dt_t_plot = ax[0, 0].imshow(Dt_truth, cmap='gray', clim=(0, 0.003))
    ax[0, 0].set_title('Dt, ground truth')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    fig.colorbar(Dt_t_plot, ax=ax[0, 0], fraction=0.046, pad=0.04)

    Dt_plot = ax[1, 0].imshow(np.reshape(paramsNN[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
    ax[1, 0].set_title('Dt, estimate')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[1, 0], fraction=0.046, pad=0.04)

    coefov = paramsSTD[0] / paramsNN[0]
    Dt_plot = ax[2, 0].imshow(np.reshape(coefov, (sx, sy)), cmap='viridis', clim=(np.min(coefov), np.max(coefov)))
    ax[2, 0].set_title('Dt, coef of var')
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[2, 0], fraction=0.046, pad=0.04)

    Dt_plot = ax[3, 0].imshow(np.reshape(paramsSTD[0], (sx, sy)), cmap='viridis')
    ax[3, 0].set_title('Dt, stdev')
    ax[3, 0].set_xticks([])
    ax[3, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[3, 0], fraction=0.046, pad=0.04)

    if arg.fit.do_fit:
        Dt_fit_plot = ax[4, 0].imshow(np.reshape(paramsf[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
        ax[4, 0].set_title('Dt, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[4, 0].set_xticks([])
        ax[4, 0].set_yticks([])
        fig.colorbar(Dt_fit_plot, ax=ax[4, 0], fraction=0.046, pad=0.04)

    Fp_t_plot = ax[0, 1].imshow(Fp_truth, cmap='gray', clim=(0, 0.5))
    ax[0, 1].set_title('Fp, ground truth')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.colorbar(Fp_t_plot, ax=ax[0, 1], fraction=0.046, pad=0.04)

    Fp_plot = ax[1, 1].imshow(np.reshape(paramsNN[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
    ax[1, 1].set_title('Fp, estimate')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[1, 1], fraction=0.046, pad=0.04)

    coefov = paramsSTD[1] / paramsNN[1]
    Dt_plot = ax[2, 1].imshow(np.reshape(coefov, (sx, sy)), cmap='viridis', clim=(np.min(coefov), np.max(coefov)))
    ax[2, 1].set_title('Fp, coef of var')
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[2, 1], fraction=0.046, pad=0.04)
    
    Dt_plot = ax[3, 1].imshow(np.reshape(paramsSTD[1], (sx, sy)), cmap='viridis')
    ax[3, 1].set_title('Fp, stdev')
    ax[3, 1].set_xticks([])
    ax[3, 1].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[3, 1], fraction=0.046, pad=0.04)
    
    if arg.fit.do_fit:
        Fp_fit_plot = ax[4, 1].imshow(np.reshape(paramsf[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
        ax[4, 1].set_title('f, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[4, 1].set_xticks([])
        ax[4, 1].set_yticks([])
        fig.colorbar(Fp_fit_plot, ax=ax[4, 1], fraction=0.046, pad=0.04)



    Dp_t_plot = ax[0, 2].imshow(Dp_truth, cmap='gray', clim=(0.01, 0.1))
    ax[0, 2].set_title('Dp, ground truth')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    fig.colorbar(Dp_t_plot, ax=ax[0, 2], fraction=0.046, pad=0.04)

    Dp_plot = ax[1, 2].imshow(np.reshape(paramsNN[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
    ax[1, 2].set_title('Dp, estimate')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(Dp_plot, ax=ax[1, 2], fraction=0.046, pad=0.04)

    coefov = paramsSTD[2] / paramsNN[2]
    Dt_plot = ax[2, 2].imshow(np.reshape(coefov, (sx, sy)), cmap='viridis', clim=(np.min(coefov), np.max(coefov)))
    ax[2, 2].set_title('Dp, coef of var')
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[2, 2], fraction=0.046, pad=0.04)
    
    Dt_plot = ax[3, 2].imshow(np.reshape(paramsSTD[2], (sx, sy)), cmap='viridis')
    ax[3, 2].set_title('Dp, stdev')
    ax[3, 2].set_xticks([])
    ax[3, 2].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[3, 2], fraction=0.046, pad=0.04)
    
    if arg.fit.do_fit:
        Dp_fit_plot = ax[4, 2].imshow(np.reshape(paramsf[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
        ax[4, 2].set_title('Dp, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[4, 2].set_xticks([])
        ax[4, 2].set_yticks([])
        fig.colorbar(Dp_fit_plot, ax=ax[4, 2], fraction=0.046, pad=0.04)

        plt.subplots_adjust(hspace=0.2)
        # plt.show()
    plt.savefig(f'{path}.png')



arg = hp_example_1()
arg = deep.checkarg(arg)
net_params = deep_simpl.net_params()

SAMPLES = 50
SNR = 15
EPOCHS = 1000

IVIM_signal_noisy, D, f, Dp = sim.sim_signal(SNR, arg.sim.bvalues, sims=arg.sim.sims, Dmin=arg.sim.range[0][0],
                                            Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                            fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                            Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)
bvalues = torch.FloatTensor(arg.sim.bvalues[:]).to(arg.train_pars.device)

for dropout in [0.1, 0.25, 0.5, 0.75]:
    # Train network.
    net_params.dropout = dropout
    net = deep_simpl.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg, epochs=EPOCHS, net_params=net_params)
    MODEL_PATH = f"./mcd-p-exploration/SNR-{SNR}_drouput-{dropout}"
    torch.save(net, f"{MODEL_PATH}.pt")

    # Simulate IVIM signal for prediction.
    [dwi_image_long, Dt_truth, Fp_truth, Dp_truth] = sim.sim_signal_predict(arg, SNR)

    # Predict.
    params = defaultdict(list)
    for i in range(SAMPLES):
        Dt, Fp, Ds, f0 = deep_simpl.predict_IVIM(dwi_image_long, arg.sim.bvalues, net, arg)
        params['Dt'].append(Dt)
        params['Fp'].append(Fp)
        params['Ds'].append(Ds)
        params['f0'].append(f0)
    
    # Print stats.
    with open(f"{MODEL_PATH}.stats", 'w') as fd:
        for pp in 'Dt Fp Ds f0'.split():
            std = np.std(params[pp], axis=1)
            mean = np.mean(params[pp], axis=1)
            coef_of_var = std / mean

            fd.write(f"\t{pp}: Stdev: (mean={np.mean(std):.6f}, med={np.median(std):.6f}, min={np.min(std):.6f}, max={np.max(std):.6f})\n")
            fd.write(f"\t    CofV:  (mean={np.mean(coef_of_var):.6f}, med={np.median(coef_of_var):.6f}, min={np.min(coef_of_var):.6f}, max={np.max(coef_of_var):.6f})\n")

    # Plot results.
    paramsNN = np.mean([params[x] for x in 'Dt Fp Ds f0'.split()], axis=1)
    paramsf = fit.fit_dats(arg.sim.bvalues, dwi_image_long, arg.fit)
    plot_example4(params, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, MODEL_PATH)