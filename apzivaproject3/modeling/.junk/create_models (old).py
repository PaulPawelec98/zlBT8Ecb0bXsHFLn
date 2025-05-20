# %% Packages

# paths
import os
import sys
from pathlib import Path

# data
import pandas as pd
# import numpy as np
import pickle
import random

# ranking
from mittens import GloVe
import fasttext
import torch
# import torch.nn.functional as f
from sentence_transformers import SentenceTransformer
# from huggingface_hub import logging

# ranking
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# seed settings
random.seed(1)
torch.manual_seed(1)

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        user_variables
        )

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Setup

# Load Data -------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
co_occurance = pd.read_csv(PROCESSED_DATA_DIR / "co_occurance_matrix.csv")
co_occurance.set_index(co_occurance.columns[0], inplace=True)
co_occurance_array = co_occurance.to_numpy()

# with open(PROCESSED_DATA_DIR / "corpus.txt", 'r') as f:
#     corpus = f.read()
# -----------------------------------------------------------------------------

# variables -------------------------------------------------------------------
if user_variables['target_string'] is None:
    target_string = 'aspiring human resources'
else:
    target_string = user_variables['target_string']

if user_variables['penalty_string'] is None:
    penalty_string = "data analyst"
else:
    target_string = user_variables['penalty_string']
# -----------------------------------------------------------------------------
'''
Set default variables if not changed.
'''

# split string ----------------------------------------------------------------
df['job_title_list'] = [
    sublist.split(" ") for sublist in df['job_title_string']
    ]
# -----------------------------------------------------------------------------

# %% GloVe

# GloVe -----------------------------------------------------------------------

# cooccurrence_matrix
model = GloVe(n=50, max_iter=1000)  # 50-dimensional embeddings

# train
word_vectors = model.fit(co_occurance_array)

# dump model
with open(MODELS_DIR / 'myglove/my_glove_model.pkl', 'wb') as file:
    pickle.dump(word_vectors, file)
# -----------------------------------------------------------------------------

# %% fasttext

# fasttext --------------------------------------------------------------------
model = fasttext.train_unsupervised(
    str(PROCESSED_DATA_DIR / "corpus.txt"),
    model='skipgram'
    )

model.save_model(str(MODELS_DIR / 'myfasttext/my_fasttext_model.bin'))
# -----------------------------------------------------------------------------

# %% BERT

# BERT ------------------------------------------------------------------------
tokenizer = torch.hub.load(
    'huggingface/pytorch-transformers',
    'tokenizer',
    'bert-base-uncased'
    )

model = torch.hub.load(
    'huggingface/pytorch-transformers',
    'model',
    'bert-base-uncased'
    )

sentences = df['job_title_string'].tolist().copy()
sentences.append(target_string)

# setup model
# device = torch.device("cpu")
model = model.eval()  # no training
# model = model.to(device)  # send to device


encodings = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors='pt',
    )

# encodings = encodings.to(device)

# disable gradient calculations
with torch.no_grad():
    # get the model embeddings
    embeds = model(**encodings)

# export
torch.save(embeds, str(MODELS_DIR / "mybert/mybertembeddings.pt"))
# -----------------------------------------------------------------------------

# BERT with two outputs -------------------------------------------------------
# sentences = df['job_title_string'].tolist().copy()

# encodings1 = tokenizer(
#     sentences,
#     padding=True,
#     truncation=True,
#     return_tensors='pt',
#     )

# encodings2 = tokenizer(
#     target_string,
#     padding=True,
#     truncation=True,
#     return_tensors='pt',
#     )

# # disable gradient calculations
# with torch.no_grad():
#     # get the model embeddings
#     embeds1 = model(**encodings1)
#     embeds2 = model(**encodings2)

# # export
# torch.save(embeds1, str(MODELS_DIR / "mybert/mybertembeddings1.pt"))
# torch.save(embeds2, str(MODELS_DIR / "mybert/mybertembeddings2.pt"))
# -----------------------------------------------------------------------------

# %% SBERT

# SBERT -----------------------------------------------------------------------
# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = df['job_title_string'].tolist().copy()
sentences.append(target_string)

# Calculate embeddings by calling model.encode()
embeddings_sbert = model.encode(sentences)

# torch.save(embeddings_sbert, str(MODELS_DIR / "mybert/mysbertembeddings.pt"))
# -----------------------------------------------------------------------------

# %% SBERT w/ penalty

# SBERT -----------------------------------------------------------------------
# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = df['job_title_string'].tolist().copy()
sentences.append(target_string)
sentences.append(penalty_string)

# Calculate embeddings by calling model.encode()
embeddings_sbert_wpen = model.encode(sentences)

# torch.save(
#     embeddings_sbert_wpen, str(
#         MODELS_DIR / "mybert/mysbertembeddingswpenalty.pt"
#         )
# )
# -----------------------------------------------------------------------------

# %% RankNet

# ranking data
rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")


# Split the data into training and test sets (80% train, 20% test)
X = rankdata.loc[:, rankdata.columns[:-1]]
y = rankdata['rank']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
    )

# To Tensor
tensor = torch.tensor(X_train.values)  # to tensor
tensor = tensor.float()  # Convert to float32

'''
Take these vectors that already show user preferences.
'''


class PairwiseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = y.values
        self.pairs = []

        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    label = 1.0 if self.y[i] > self.y[j] else 0.0
                    self.pairs.append((i, j, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        x1 = self.X[i]
        x2 = self.X[j]
        return x1, x2, torch.tensor([label], dtype=torch.float32)


pairwise_dataset = PairwiseDataset(X_train, y_train)

loader = DataLoader(
    pairwise_dataset, batch_size=32, shuffle=True, num_workers=4
    )


# ranknet ---------------------------------------------------------------------
class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
        # first item
        x1_in = self.activation(self.input(x1))
        h1 = self.activation(self.hidden(x1_in))
        out1 = self.output(h1)

        # second item
        x2_in = self.activation(self.input(x2))
        h2 = self.activation(self.hidden(x2_in))
        out2 = self.output(h2)

        # difference
        diff = out1 - out2
        return diff


# variables
input_size = len(rankdata.iloc[0, :-1])  # size of the input vector
hidden_size = 16
learning_rate = 0.001
model = RankNet(input_size, hidden_size)
epoch_range = 50

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # applies sigmoid
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example usage
target = torch.tensor([[1.0]])
train_loss = []

# Train Model -----------------------------------------------------------------
for epoch in range(epoch_range):
    model.train()
    for x1, x2, label in loader:
        optimizer.zero_grad()
        diff = model(x1, x2)
        loss = criterion(diff, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
# -----------------------------------------------------------------------------

# for epoch in range(epoch_range):

#     for i in range(0, len(X_train), 1):

#         x1_tensor = tensor[i, :].unsqueeze(0)

#         for j in range(0, len(X_train), 1):

#             x2_tensor = tensor[j, :].unsqueeze(0)

#             model.train()

#             # clear gradients
#             optimizer.zero_grad()

#             # Forward pass
#             diff = model(x1_tensor, x2_tensor)

#             # Compute loss
#             loss = criterion(diff, target)

#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()

#             train_loss.append(loss.item())

#             print(i, j)

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss.item()}")
            #     with torch.no_grad():
            #         if diff.item() > 0:  # check which ranks better
            #             print(
            #                 f"""Epoch {epoch},
            #                 Output: {diff.item()} - First input rank higher."""
            #                 )
            #         else:
            #             print(
            #                 f"""Epoch {epoch},
            #                 Output: {diff.item()} - Second input rank higher."""
            #                 )
        # -----------------------------------------------------------------------------
