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

# Load Model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small",
    device_map="auto",
    torch_dtype=torch.float16  # Reduce memory
)

# Test ------------------------------------------------------------------------
# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
# -----------------------------------------------------------------------------



# input_text = (
#     f"Given the keywords '{target_string}', which of the following job titles \
# is more relevant?\n\n"
#     f"1. {df.iloc[0, 0]}\n"
#     f"2. {df.iloc[1, 0]}\n\n"
#     f"Answer with '1' or '2'."
#     )

# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))



# input_text1 = (
#     f"Given the keywords '{target_string}', which of the following job titles \
# is more relevant?\n\n"
#     f"1. {df.iloc[0, 0]}\n"
#     f"2. {df.iloc[1, 0]}\n\n"
#     f"Answer with '1' or '2'."
#     )

# input_text2 = (
#     f"Given the keywords '{target_string}', which of the following job titles \
# is more relevant?\n\n"
#     f"1. {df.iloc[9, 0]}\n"
#     f"2. {df.iloc[2, 0]}\n\n"
#     f"Answer with '1' or '2'."
#     )


# input_ids = tokenizer(
#     [input_text1, input_text2], return_tensors="pt", padding=True
#     ).input_ids.to("cuda")

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[1]))
# -----------------------------------------------------------------------------

# scores = {idx: 0 for idx in df.index.tolist()}

# for text1, text2, i, j in rankloader:

#     prompts = []
#     for k in range(0, len(text1), 1):

#         texti = text1[k]
#         textj = text2[k]

#         prompt = (
#             f"Given the keywords '{target_string}',\
# which of the following job titles \
# is more relevant?\n"
#             f"1. {texti}\n"
#             f"2. {textj}\n"
#             f"Answer with '1' or '2'."
#             )
#         prompts.append(prompt)

#     input_ids = tokenizer(
#         prompts, return_tensors="pt", padding="longest", truncation=True
#         ).input_ids.to("cuda")

#     outputs = model.generate(input_ids, max_new_tokens=1)

#     for k in range(0, len(text1), 1):
#         out = tokenizer.decode(outputs[k])
#         # print(out)
#         try:
#             digit = int(re.search(r'\d+', out).group(0))
#         except AttributeError:
#             print(f"No digit found in output: {out!r}")
#             continue  # Must be inside a loop

#         if digit == 1:
#             scores[int(i[k])] += 1
#         else:
#             scores[int(j[k])] += 1
#         print(i, j, digit)


# torch.save(all_prompts, INTERIM_DATA_DIR / 'flan_prompt_tokens.pt')


# Premade Token Prompts -------------------------------------------------------
flan_job_tensors = torch.load(INTERIM_DATA_DIR / 'flan_job_tensors.pt')

list_of_prompts = [
    f"Given the keywords '{target_string}',\
    which of the following job titles \
    is more relevant?\n",
    "1.",
    "2.",
    "Answer with '1' or '2'."
    ]

remove_seps = [0, 1, 1, 0]

list_of_inputs = [
    generate_tokens(input_id, tokenizer) for input_id in list_of_prompts
    ]

list_of_inputs = [
    remove_seperator(input_id)
    if remove_sep == 1
    else input_id
    for input_id, remove_sep
    in zip(list_of_inputs, remove_seps)
    ]

tokenpairs = TokenPairs(flan_job_tensors)
tokenloaders = DataLoader(tokenpairs, batch_size=1)

all_prompts = []
concat_idx = [1, 2]

for pair in tokenloaders:
    inputs = list_of_inputs.copy()
    inputs[concat_idx[0]],  inputs[concat_idx[1]] = (
        torch.cat([inputs[concat_idx[0]], pair[0]], dim=1),
        torch.cat([inputs[concat_idx[1]], pair[1]], dim=1)
        )
    inputs = torch.cat(inputs, dim=1)
    all_prompts.append(inputs)
    print(int(pair[2]), int(pair[3]))

rankdataset = RankDatasetPairs(df)
rankloader = DataLoader(
        rankdataset, batch_size=100, shuffle=False, num_workers=0
        )
rankiter = iter(rankloader)
next(rankiter)
# -----------------------------------------------------------------------------


scores = {idx: 0 for idx in df.index.tolist()}

for text1, text2, i, j in rankloader:

    prompts = []
    for k in range(0, len(text1), 1):

        texti = text1[k]
        textj = text2[k]

    input_ids = tokenizer(
        prompts, return_tensors="pt", padding="longest", truncation=True
        ).input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens=1)

    for k in range(0, len(text1), 1):
        out = tokenizer.decode(outputs[k])
        # print(out)
        try:
            digit = int(re.search(r'\d+', out).group(0))
        except AttributeError:
            print(f"No digit found in output: {out!r}")
            continue  # Must be inside a loop

        if digit == 1:
            scores[int(i[k])] += 1
        else:
            scores[int(j[k])] += 1
        print(i, j, digit)
