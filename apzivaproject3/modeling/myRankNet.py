# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 17:27:43 2025

@author: Paul
"""

# %% setup
# Date
# from datetime import datetime

# My Pairwise RankNet Model
import torch
from torch import nn
from torch.utils.data import Dataset
import itertools
# from sklearn.model_selection import train_test_split

# Setup -----------------------------------------------------------------------
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        # PROCESSED_DATA_DIR,
        # MODELS_DIR,
        MODELING_DIR
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% RankNet Model


class RankDataset(Dataset):
    # For normal ranknet
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RankDatasetStars(Dataset):
    # for ranknet with pairs
    def __init__(self, X, y, star):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.star = torch.tensor(star.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.star[idx]


class RankDatasetPairs(Dataset):
    # to predict with all possible pairs already made.
    def __init__(self, data):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.num_samples = len(data)
        self.pairs = list(itertools.combinations(range(self.num_samples), 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.data[i], self.data[j], i, j


class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        print(f"""Initializing RankNet with input_size={input_size},
              hidden_size={hidden_size}"""
              )
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()  # cuts negative scores.

    def forward(self, x1, x2):
        h1 = self.activation(self.hidden(self.activation(self.input(x1))))
        h2 = self.activation(self.hidden(self.activation(self.input(x2))))
        return self.output(h1) - self.output(h2)
