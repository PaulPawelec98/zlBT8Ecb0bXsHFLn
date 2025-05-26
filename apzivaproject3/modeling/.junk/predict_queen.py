# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 21:41:16 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
from pathlib import Path
import pandas as pd

# models
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
    )

# from transformers.utils import default_cache_path

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        CACHE_DIR
        )

    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ['HF_HOME'] = str(CACHE_DIR)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------


# %% Load Queen Model

# Load Model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Bytes512/Queen")
model = AutoModelForCausalLM.from_pretrained("Bytes512/Queen")