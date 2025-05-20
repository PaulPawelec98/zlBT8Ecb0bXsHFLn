# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:10:58 2025

@author: Paul
"""

# %% Co-occurance Matrix

# import nltk
from collections import defaultdict, Counter
import numpy as np
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
        RESOURCES_DIR
        )

    import nltk
    nltk.data.path.append(str(RESOURCES_DIR))

    from nltk.tokenize import word_tokenize


except Exception as e:
    print(f"Error importing config/nltk: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------


def create_co_matrix(text, window_size=1):

    if not text:
        # Sample text
        text = """Apple is looking at buying U.K. startup for $1 billion. The
        deal is expected to close by January 2022. Apple is very optimistic
        about the acquisition."""

    # # Download NLTK resources
    # nltk.download('punkt_tab', download_dir=str(RESOURCES_DIR))
    # nltk.download('punkt', download_dir=str(RESOURCES_DIR))
    # nltk.download('stopwords', download_dir=str(RESOURCES_DIR))

    # Preprocess the text
    words = word_tokenize(text.lower())

    # Define the window size for co-occurrence
    window_size = 3

    # Create a list of co-occurring word pairs
    co_occurrences = defaultdict(Counter)
    for i, word in enumerate(words):
        for j in range(
                max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                co_occurrences[word][words[j]] += 1

    # Create a list of unique words
    unique_words = list(set(words))

    # Initialize the co-occurrence matrix
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

    # Populate the co-occurrence matrix
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    # Create a DataFrame for better readability
    co_matrix_df = pd.DataFrame(
        co_matrix, index=unique_words, columns=unique_words
        )

    return co_matrix_df
