from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import time
import torch
import sys

from hyperparams import hyperparams as hp_example_1
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
import IVIMNET.deep_simplified as deep_simpl
import IVIMNET.fitting_algorithms as fit


arg = hp_example_1()
arg = deep.checkarg(arg)
net_params = deep_simpl.net_params()
net_params.dropout = 0.125

MODEL = "ds1_avg"
SAMPLES = 50
EPOCHS = 1000

FOLDER = "./invivo_data"
DATASET = f"{FOLDER}/{MODEL}"
MODEL_PATH = f"./models/invivo/{MODEL}"

bvalues = np.genfromtxt(f"{DATASET}.bval")
selsb = np.array(bvalues) == 0

data = nib.load(f"{DATASET}.nii")
datas = data.get_fdata()
sx, sy, sz, n_b_values = datas.shape
X_dw = np.reshape(datas, (sx * sy * sz, n_b_values))

### select only relevant values, delete background and noise, and normalise data
S0 = np.nanmean(X_dw[:, selsb], axis=1)
S0[S0 != S0] = 0
S0 = np.squeeze(S0)
valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) 
datatot = X_dw[valid_id, :]
# normalise data
S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
datatot = datatot / S0[:, None]

res = [i for i, val in enumerate(datatot != datatot) if not val.any()] # Remove NaN data
# net = deep_simpl.learn_IVIM(datatot[res], bvalues, arg, epochs=EPOCHS, net_params=net_params)
# torch.save(net, f"{MODEL_PATH}.pt")
net = torch.load(f"{MODEL_PATH}.pt")

# Predict.
params = defaultdict(list)
for i in range(SAMPLES):
    Dt, Fp, Ds, f0 = deep_simpl.predict_IVIM(datatot, bvalues, net, arg)
    params['Dt'].append(Dt)
    params['Fp'].append(Fp)
    params['Ds'].append(Ds)
    params['f0'].append(f0)

# Write out images.
for name in "Dt Fp Ds f0".split():
    img = np.zeros([sx * sy * sz])
    img[valid_id] = np.std(params[name], axis=0)[:sum(valid_id)]
    img = np.reshape(img, [sx, sy, sz])
    nib.save(nib.Nifti1Image(img, data.affine, data.header), f"{MODEL_PATH}-{name}-stdev.nii")