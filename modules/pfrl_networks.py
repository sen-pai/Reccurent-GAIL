import torch
import torch.nn as nn

import pfrl
from pfrl.policies import SoftmaxCategoricalHead


def get_mlp_model(obs_size, n_actions):
    mlp_model = nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.ReLU(),
        pfrl.nn.Branched(
            # action branch
            nn.Sequential(nn.Linear(64, n_actions), SoftmaxCategoricalHead(),),
            # value branch
            nn.Linear(64, 1),
        ),
    )
    return mlp_model


def get_rnn_model(obs_size, n_actions):
    rnn_model = pfrl.nn.RecurrentSequential(
        nn.Linear(obs_size, 64),
        nn.ReLU(),
        nn.GRU(num_layers=1, input_size=64, hidden_size=64),
        pfrl.nn.Branched(
            # action branch
            nn.Sequential(nn.Linear(64, n_actions), SoftmaxCategoricalHead(),),
            # value branch
            nn.Linear(64, 1),
        ),
    )
    return rnn_model
