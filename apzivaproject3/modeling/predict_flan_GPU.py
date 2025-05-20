# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:41:12 2025

@author: Paul
"""

# %% Setup

# Data
import pandas as pd
import re
import time
import math
from collections import deque

# Transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# torch
import torch
from torch.utils.data import DataLoader


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

    import myLLMs

except Exception as e:
    print(f"Error importing config: {e}")
# -----------------------------------------------------------------------------

# %% flan-small


# Timing Stuff

total_pairs = 1254 * (1254 - 1) // 2  # 785631
batch_size = 100
total_batches = math.ceil(total_pairs / batch_size)

batch_times = deque(maxlen=5)  # Rolling window of the last 5 batch times
overall_start = time.time()


def predict_flan():
    target_string = "aspiring human resources"
    df = pd.read_csv(PROCESSED_DATA_DIR / 'clean_df.csv')

    # Load Flan Model ---------------------------------------------------------
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-small",
        device_map="auto",
        torch_dtype=torch.float16  # Reduce memory
    )
    # -------------------------------------------------------------------------

    # Premade Token Prompts ---------------------------------------------------
    flan_job_tensors = torch.load(INTERIM_DATA_DIR / 'flan_job_tensors.pt')

    list_of_prompts = [
        f"Given the keywords '{target_string}',\
        which of the following job titles \
        is more relevant?\n",
        "1.",
        "2.",
        "Answer with '1' or '2'."
        ]

    remove_seps = [0, 1, 1, 0]  # which tokens to remove <\s> from

    list_of_inputs = [
        myLLMs.generate_tokens(
            input_id, tokenizer
            ) for input_id
        in list_of_prompts
        ]

    list_of_inputs = [
        myLLMs.remove_seperator(input_id)
        if remove_sep == 1
        else input_id
        for input_id, remove_sep
        in zip(list_of_inputs, remove_seps)
        ]

    concat_idx = [1, 2]  # Which indexes to concat with new job tokens

    tokenpairs = myLLMs.TokenPairs(
        flan_job_tensors, list_of_inputs, concat_idx
        )

    tokenloaders = DataLoader(
        tokenpairs, batch_size=100, num_workers=2, collate_fn=myLLMs.collate_fn
        )

    scores = {idx: 0 for idx in df.index.tolist()}

    for batch_idx, (prompts, i, j) in enumerate(tokenloaders):
        batch_start = time.time()

        # ----------------- Inference ------------------
        outputs = model.generate(prompts.to("cuda"), max_new_tokens=1)

        # ----------------- Post-processing ------------------
        for k in range(len(prompts)):
            out = tokenizer.decode(outputs[k])
            print(out)
            try:
                digit = int(re.search(r'\d+', out).group(0))
            except AttributeError:
                print(f"No digit found in output: {out!r}")
                continue

            if digit == 1:
                scores[int(i[k])] += 1
            else:
                scores[int(j[k])] += 1

        batch_end = time.time()
        elapsed = batch_end - batch_start
        batch_times.append(elapsed)

        avg_time = sum(batch_times) / len(batch_times)
        batches_left = total_batches - batch_idx
        eta = avg_time * batches_left

        print(
            f"\n Batch {batch_idx}/{total_batches} | "
            f"Batch Time: {elapsed:.2f}s | "
            f"ETA: {eta / 60:.2f} minutes ({eta:.1f} seconds)\n"
        )

    print(f" All done in {time.time() - overall_start:.2f} seconds")

    return scores


# %% Name = Main

if __name__ == "__main__":
    scores = predict_flan()
    df = pd.DataFrame(list(scores.items()), columns=["id", "score"])
    df.to_csv(INTERIM_DATA_DIR / "flan_scores.csv", index=False)
