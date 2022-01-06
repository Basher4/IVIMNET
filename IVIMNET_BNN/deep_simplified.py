from tqdm import tqdm
from typing import Optional
from collections import namedtuple
from dataclasses import dataclass
import numpy as np
import copy
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchbnn as bnn
from torchbnn.modules.batchnorm import _BayesBatchNorm

import IVIMNET.fitting_algorithms as fit

MinMax = namedtuple('MinMax', ['min', 'max'])

@dataclass
class MinMax:
    min: float
    max: float
    def delta(self):
        return self.max - self.min

@dataclass
class ParamBounds:
    D:  MinMax
    f:  MinMax
    Dp: MinMax
    f0: MinMax
    Dp2: MinMax = None
    f2: MinMax = None

@dataclass
class net_params:
    """ Replicating 'optim' settings from hyperparameters.py """
    dropout:    Optional[float] = 0.1
    batch_norm: bool    = True
    tri_exp:    bool    = False
    fitS0:      bool    = True
    depth:      int     = 2
    width:      Optional[int] = None  # Wide as number of b-values
    parallel:   str     = 'parallel'
    con:        str     = 'sigmoid'
    bounds:     ParamBounds = ParamBounds(D  = MinMax(0.0, 0.005), f  = MinMax(0.0, 0.7), # Dt, Fp
                                          Dp = MinMax(0.005, 0.2), f0 = MinMax(0.0, 2.0)) # Ds, S0

class BayesBatchNorm1d(_BayesBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )

class Net(nn.Module):
    def __init__(self, bvalues: np.array, net_params: net_params):
        super().__init__()

        self.bvalues = bvalues
        self.net_params = net_params
        self.net_params.width = self.net_params.width or len(bvalues)

        # Assertions I make because I don't want to implement all the option tree.
        assert(net_params.batch_norm)
        assert(net_params.fitS0)
        assert(net_params.parallel == 'parallel')
        assert(net_params.con == 'sigmoid')
        assert(not net_params.tri_exp)

        self.est_params = 5 if net_params.tri_exp else 3
        if net_params.fitS0:
            self.est_params += 1

        width = len(bvalues)
        self.fc_layers = nn.ModuleList([nn.ModuleList() for _ in range(self.est_params)])
        for i in range(self.net_params.depth):
            for layer in self.fc_layers:
                layer.extend([
                    bnn.BayesLinear(prior_mu=0, prior_sigma=10, in_features=width, out_features=self.net_params.width),
                    BayesBatchNorm1d(prior_mu=0, prior_sigma=10, num_features=self.net_params.width),
                    nn.ELU(),
                ])
                if i < self.net_params.depth - 1 and self.net_params.dropout:
                    layer.extend([nn.Dropout(self.net_params.dropout)])

        # Parallel network to estimate each parameter separately.
        self.encoder = nn.ModuleList([nn.Sequential(*fcl, bnn.BayesLinear(prior_mu=0, prior_sigma=10,
                                                                          in_features=self.net_params.width, out_features=1))
                                      for fcl in self.fc_layers])

    def forward(self, X):
        assert(self.net_params.parallel == 'parallel')
        assert(self.net_params.con == 'sigmoid')
        assert(not self.net_params.tri_exp)

        def sigm(param, bound: MinMax):
            return bound.min + torch.sigmoid(param[:, 0].unsqueeze(1)) * bound.delta()

        pb = self.net_params.bounds
        params = [enc(X) for enc in self.encoder]
        Dt = sigm(params[2], pb.D)
        Fp = sigm(params[0], pb.f)
        Dp = sigm(params[1], pb.Dp)
        f0 = sigm(params[3], pb.f0)

        # loss function
        X = Fp * torch.exp(-self.bvalues * Dp) + f0 * torch.exp(-self.bvalues * Dt)
        return X, Dt, Fp/(f0+Fp), Dp, f0+Fp

