# %% Packages

# paths
import os
import json
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
        MODELING_DIR,
        SETTINGS_FILE
        )

    os.chdir(MODELING_DIR)

    if str(MODELING_DIR) not in sys.path:
        sys.path.append(str(MODELING_DIR))

    from myRankNet import Train_RankNet_Pairwise

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way

os.chdir(PROJ_ROOT)
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
with open(SETTINGS_FILE, 'r') as f:
    settings = json.load(f)

target_string = settings['target_settings']['target_string']
penalty_string = settings['target_settings']['penalty_string']
# -----------------------------------------------------------------------------
'''
Return user set variables...
'''

print("Target is...", target_string)
print("Penalty is...", penalty_string)

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

torch.save(embeddings_sbert, str(MODELS_DIR / "mybert/mysbertembeddings.pt"))
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

torch.save(
    embeddings_sbert_wpen, str(
        MODELS_DIR / "mybert/mysbertembeddingswpenalty.pt"
        )
)
# -----------------------------------------------------------------------------

# %% RankNet

# rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")
# Train_RankNet_Pairwise(rankdata)  # exports to models/myranknet/myranknet.pt
