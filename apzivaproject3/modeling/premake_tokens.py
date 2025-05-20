# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:53:59 2025

@author: Paul
"""

# %% Setup

# data
import pandas as pd

# torch
import torch

# Transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        PROCESSED_DATA_DIR,
        MODELING_DIR,
        INTERIM_DATA_DIR
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------


# %% Data

# data ------------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
jobs = df['job_title_string'].tolist()
# -----------------------------------------------------------------------------

# %% Flan-Tokenizers

# flan-tokenizer --------------------------------------------------------------

tokenizer = T5Tokenizer.from_pretrained(
    "google/flan-t5-small"
    )

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small",
    device_map="auto",
    torch_dtype=torch.float16  # Reduce memory
)

job_tokens = []

for job in jobs:
    input_ids = tokenizer(
        job, return_tensors="pt", padding="longest", truncation=True
        ).input_ids.to("cuda")
    job_tokens.append(input_ids)

torch.save(job_tokens, INTERIM_DATA_DIR / 'flan_job_tensors.pt')
# loaded_list = torch.load('tensor_list.pt')
# -----------------------------------------------------------------------------