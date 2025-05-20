# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 17:04:27 2025

@author: Paul
"""
# %% setup

# Data
import pandas as pd
import numpy as np

# Date
from datetime import datetime

# Torch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Setup -----------------------------------------------------------------------
import os
import sys
import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        MODELING_DIR,
        SETTINGS_FILE
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

    os.chdir(MODELING_DIR)
    import myRankNet
    os.chdir(PROJ_ROOT)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------


# %% RankNet Predict

# ranknet predict -------------------------------------------------------------


def predict_ranknet_star():

    # rankdata
    rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")
    df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

    # variables...
    input_size = 7  # from training
    hidden_size = 64

    # with open(SETTINGS_FILE, 'r') as f:
    #     settings = json.load(f)

    # Recreate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myRankNet.RankNet(input_size, hidden_size)
    model.load_state_dict(torch.load(
        MODELS_DIR / "myranknet/myranknet.pt")
        )
    model.to(device)

    # Get Model Results
    model.eval()

    # Load and Setup Data
    X = rankdata.loc[:, rankdata.columns[:-2]]

    # Add In Starred Candidates...
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)

    star_vec = np.zeros(len(X), dtype=int)
    star_vec[settings['starred_candidates']] = 1

    X['star'] = star_vec

    # y = rankdata['rank']

    rankthis = myRankNet.RankDatasetPairs(X)

    rankthisloader = DataLoader(
            rankthis, batch_size=256, shuffle=False, num_workers=2
            )

    scores = {index: 0 for index in X.index}

    with torch.no_grad():
        for pair in rankthisloader:
            x1, x2 = pair[0], pair[1]
            x1, x2 = x1.to(device), x2.to(device)

            i, j = pair[2], pair[3]
            print(i, j)

            diff = model(x1, x2)

            # Loop through the batch and update the scores
            for k in range(x1.size(0)):  # x1.size(0) gives batch_size
                if diff[k].item() > 0:
                    scores[int(i[k])] += 1
                else:
                    scores[int(j[k])] += 1

    # Add scores to dataframe and rank
    rankdata["score"] = scores.values()
    rankdata['job_title_string'] = df['job_title_string']

    print(rankdata[["score", "rank"]].head())

    # parameters...
    for name, param in model.named_parameters():
        print(f"{name}:\n{param.data}\n")

    return rankdata

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    rankdata = predict_ranknet_star()
    rankdata.to_csv(PROCESSED_DATA_DIR / 'df_with_rank.csv', index=False)
