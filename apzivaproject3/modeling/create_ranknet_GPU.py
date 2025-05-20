# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:02:34 2025

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

# %% Train MyRankNet


def Train_RankNet_Pairwise(df):
    # Copy data
    rankdata = df.copy()

    # Split data
    X = rankdata.loc[:, rankdata.columns[:-2]]
    y = rankdata["rank"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Check if GPU is available
    device = torch.device("cuda")
    print(f"CUDA?: {torch.cuda.is_available()}")

    # Dataset and DataLoader
    train_dataset = myRankNet.RankDatasetStars(X_train, y_train)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=8
        )

    # Model setup
    input_size = X.shape[1]
    hidden_size = 32

    model = myRankNet.RankNet(input_size, hidden_size).to(device)  # send

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # combined Sigmoid layer and BCELoss
    epoch_range = 5
    random_range = 500

    # Training loop
    for epoch in range(epoch_range):

        train_iter = iter(train_loader)  # shuffles the data.
        model.train().to(device)

        for rand in range(0, random_range, 1):

            try:
                x1, y1 = next(train_iter)
                x2, y2 = next(train_iter)
            except StopIteration:
                print("Train iterator exhausted. Breaking early.")
                break

            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            optimizer.zero_grad()

            diff = model(x1, x2)

            target = torch.tensor([[1.0]] if y1 > y2 else [[0.0]]).to(device)

            loss = criterion(diff, target)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the trained model
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = MODELS_DIR / f"myranknet/myranknet_{date_str}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# %% Train RankNet Stars


def Train_RankNet_Pairwise_Stars(df):
    # Copy data
    rankdata = df.copy()

    # Split data
    X = rankdata.loc[:, rankdata.columns[:-2]]
    y = rankdata["rank"]

    # Add In Starred Candidates...
    star_vec = np.zeros(len(X), dtype=int)

    # Define the percentage of 1's you want to set
    percentage = 0.55

    # Calculate how many elements should be set to 1
    num_ones = int(len(star_vec) * percentage)

    # Randomly select indices to set to 1
    indices_to_set = np.random.choice(
        len(star_vec), size=num_ones, replace=False
        )

    # Set those indices to 1
    star_vec[indices_to_set] = 1

    X['star'] = star_vec

    # Create Sets...
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    star_vec_pop = X_train['star'].copy()

    # Check if GPU is available
    device = torch.device("cuda")
    print(f"CUDA?: {torch.cuda.is_available()}")

    # Dataset and DataLoader
    train_dataset = myRankNet.RankDatasetStars(X_train, y_train, star_vec_pop)

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
        )

    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)

    # Model setup
    input_size = X_train.shape[1]
    hidden_size = settings['ranknet']['hidden_size']

    model = myRankNet.RankNet(input_size, hidden_size).to(device)  # send

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # combined Sigmoid layer and BCELoss
    epoch_range = 15
    random_range = 100

    # Training loop
    for epoch in range(epoch_range):

        train_iter = iter(train_loader)  # shuffles the data.
        model.train().to(device)

        for rand in range(0, random_range, 1):

            try:
                x1, y1, star1 = next(train_iter)
                x2, y2, star2 = next(train_iter)
            except StopIteration:
                print("Train iterator exhausted. Breaking early.")
                break

            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            star1, star2 = star1.to(device), star2.to(device)

            optimizer.zero_grad()

            diff = model(x1, x2)

            # let's ignore comparing against two starred candidates.
            if star1 == 1 and star2 == 1:
                continue

            target = torch.tensor(
                [[1.0]]
                if (y1 - (y1 * star2)) > (y2 - (y2 * star1))
                else [[0.0]]
                ).to(device)

            loss = criterion(diff, target)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the trained model
    # date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = f"myranknet/myranknet_{date_str}.pt"
    file_name = "myranknet/myranknet.pt"
    model_path = MODELS_DIR / file_name
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# %% Name = Main


if __name__ == "__main__":
    rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")
    Train_RankNet_Pairwise_Stars(rankdata)
