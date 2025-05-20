# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:52:07 2025

@author: Paul
"""

# %% Setup

# Data
import pandas as pd
import re

# Transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# torch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import itertools


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
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELING_DIR,
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

    import myLLMs

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

# data ------------------------------------------------------------------------
target_string = "aspiring human resouces"
df = pd.read_csv(PROCESSED_DATA_DIR / 'clean_df.csv')
rankdata = pd.read_csv(PROCESSED_DATA_DIR / 'rankdata.csv')

# RankDataSetPairs ------------------------------------------------------------


class RankDatasetPairs(Dataset):
    # to predict with all possible pairs already made.
    def __init__(self, data):
        self.texts = data["job_title_string"].tolist()
        self.num_samples = len(self.texts)
        self.pairs = list(itertools.combinations(range(self.num_samples), 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.texts[i], self.texts[j], i, j


class TokenPairs(Dataset):
    # to predict with all possible pairs already made.
    def __init__(self, tokens):
        self.tokens = tokens
        self.num_samples = len(self.tokens)
        self.pairs = list(itertools.combinations(range(self.num_samples), 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.tokens[i].squeeze(0), self.tokens[j].squeeze(0), i, j
# -----------------------------------------------------------------------------

# %% Functions

# Functions -------------------------------------------------------------------


def remove_seperator(t):
    return t[:, :-1]


def generate_tokens(prompt, tokenizer):
    input_ids = tokenizer(
        prompt, return_tensors="pt", padding="longest", truncation=True
        ).input_ids.to("cuda")
    return input_ids


# -----------------------------------------------------------------------------

# %% flan-small

# flan-small ------------------------------------------------------------------

# Load Flan Model -------------------------------------------------------------
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small",
    device_map="auto",
    torch_dtype=torch.float16  # Reduce memory
)

# Rank
# -----------------------------------------------------------------------------
