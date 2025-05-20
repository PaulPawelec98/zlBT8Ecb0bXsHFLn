# %% Packages
# =============================================================================
# Packages
# =============================================================================

import os
import json
import pandas as pd

from apzivaproject3 import (
    config,
    dataset,
    features,
    train,
    predict,
    )

from apzivaproject3.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SETTINGS_FILE,
    )

# %% Data
# =============================================================================
# Data
# =============================================================================

df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

# %% User Variables
# =============================================================================
# Set User Variables
# =============================================================================

# Inputs
target = 'data analyst'
penalty = 'scientist'

# Read User Settings
with open(SETTINGS_FILE, 'r') as f:
    settings = json.load(f)

# Update User Settings
settings['target_settings']['target_string'] = target
settings['target_settings']['penalty_string'] = penalty

# Starred Indexes
# settings['starred_candidates'] = [7]
settings['starred_candidates'] = [0, 2, 5, 7]


# settings['ranknet']['input_size'] = 7
settings['ranknet']['hidden_size'] = 64

# Write Updated Settings
with open(SETTINGS_FILE, 'w') as f:
    json.dump(settings, f, indent=4)


# %% Make Project
# =============================================================================
# Run Main Scripts
# =============================================================================

# dataset.main()
# features.main()
train.main()
predict.main()

# %% Create Documents
# =============================================================================
# Documents
# =============================================================================

df2 = pd.read_csv(PROCESSED_DATA_DIR / 'df_with_scores.csv')
df3 = pd.read_csv(PROCESSED_DATA_DIR / "df_with_rank.csv")
df4 = pd.read_csv(INTERIM_DATA_DIR / "flan_scores.csv")
df['flan_scores'] = df4['score'].copy()  # it worked!
