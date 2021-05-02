"""
File contains everything required to load, provide expert trajectories to the discriminator. 

"""

import random
import glob
import os
import math
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def pad_collate(traj_batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """

    lens = [x.shape[0] for x in traj_batch]
    padded_traj = pad_sequence(traj_batch, batch_first=True, padding_value=0)

    # labels to train disc, gen_is_high
    labels = torch.zeros(len(lens))

    return padded_traj, torch.tensor(lens, dtype=torch.float), labels


class ExpertDataset(Dataset):
    """
    dataset that stores the saved traj
    """

    def __init__(self, saved_pkl, obs_only: bool = True):

        # expert_data = [[obs_1, acts_1], [obs_2, acts_2]...]
        with open(saved_pkl, "rb") as f:
            self.expert_data = pickle.load(f)

        self.obs_only = obs_only

    def __len__(self):
        return len(self.expert_data)

    def __getitem__(self, i):
        if self.obs_only:
            traj = self.expert_data[i][0]
        else:
            # squash obs and act together
            traj = None  # will implement later

        return traj


class ExpertDataLoader:
    def __init__(
        self,
        saved_pkl,
        obs_only: bool = True,
        expert_batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset = ExpertDataset(saved_pkl, obs_only)
        self.expert_batch_size = expert_batch_size
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=expert_batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=True,
            collate_fn=pad_collate,
        )

    def get_expert_batch(self):
        return next(iter(self.dataloader))
