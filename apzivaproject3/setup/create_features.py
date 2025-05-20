# %% Setup

# data
import pandas as pd

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
        FUNCTIONS_DIR
        )

    '''
    When using subprocess, the path for FUNCTIONS_DIR, can't be found?
    I can add manually for now.
    '''

    if str(FUNCTIONS_DIR) not in sys.path:
        sys.path.append(str(FUNCTIONS_DIR))

    os.chdir(FUNCTIONS_DIR)
    print(os.getcwd())

    from create_co_occurance_matrix import create_co_matrix

    os.chdir(PROJ_ROOT)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Load Data

df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

with open(PROCESSED_DATA_DIR / "corpus.txt", 'r') as file:
    corpus = file.read()

# %% co-occurance matrix

# co-occurance matrix
co_occurance = create_co_matrix(text=corpus, window_size=3)

# %% export

co_occurance.to_csv(PROCESSED_DATA_DIR / 'co_occurance_matrix.csv', index=True)

