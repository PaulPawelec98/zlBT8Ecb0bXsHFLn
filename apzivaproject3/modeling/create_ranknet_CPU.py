# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:02:34 2025

@author: Paul
"""

# %% setup

# Data
import pandas as pd

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
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        MODELING_DIR
        )

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

    os.chdir(MODELING_DIR)
    import myRankNet
    os.chdir(PROJ_ROOT)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way

# %% Train MyRankNet


def Train_RankNet_Pairwise(df):
    # Copy data
    rankdata = df.copy()

    # Split data
    X = rankdata.loc[:, rankdata.columns[:-2]]
    y = rankdata["rank"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1
    )

    # Check if GPU is available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"CUDA?: {torch.cuda.is_available()}")

    # Dataset and DataLoader
    train_dataset = myRankNet.RankDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=3
        )

    # Model setup
    input_size = X.shape[1]
    hidden_size = 16
    model = myRankNet.RankNet(input_size, hidden_size)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # combined Sigmoid layer and BCELoss
    epoch_range = 5
    random_range = 100

    # Training loop
    for epoch in range(epoch_range):

        train_iter = iter(train_loader)
        model.train()

        for rand in range(0, random_range, 1):

            try:
                x1, y1 = next(train_iter)
                x2, y2 = next(train_iter)
            except StopIteration:
                print("Train iterator exhausted. Breaking early.")
                break

            optimizer.zero_grad()
            diff = model(x1, x2)
            target = torch.tensor([[1.0]] if y1 > y2 else [[0.0]])
            loss = criterion(diff, target)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the trained model
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = MODELS_DIR / f"myranknet/myranknet_{date_str}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")
    Train_RankNet_Pairwise(rankdata)
