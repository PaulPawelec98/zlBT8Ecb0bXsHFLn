# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:07:27 2025

@author: Paul
"""


# %% Setup

# Data
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# Embeddings
from sentence_transformers import SentenceTransformer

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
        COLLECTION_DATA_DIR
        )

except Exception as e:
    print(f"Error importing config: {e}")
# -----------------------------------------------------------------------------

# %% Data

# Load Data -------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / 'clean_df.csv')
# -----------------------------------------------------------------------------

# Prep Data for New Collection ------------------------------------------------
'''
Each chunk will just be each job title.
'''
job_tiles = df['job_title_string'].tolist()
job_ids = [f'jb_{i}' for i in range(0, len(job_tiles))]
# -----------------------------------------------------------------------------

# Prep Function for Embedding -------------------------------------------------
'''
We will use SBERT since this was the best performing model from before.
'''


class SBertEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input).tolist()


embedding_function = SBertEmbeddingFunction()

# -----------------------------------------------------------------------------

# Create chroma and setup collection ------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=str(COLLECTION_DATA_DIR)
)


# collection = chroma_client.create_collection(
#     name="job_titles_collection",
#     embedding_function=embedding_function
#     )

# collection.add(
#     documents=job_tiles,
#     ids=job_ids,
#     )
# -----------------------------------------------------------------------------

# Return Collection -----------------------------------------------------------
collection = chroma_client.get_collection(
    name="job_titles_collection",
    embedding_function=embedding_function
    )
# -----------------------------------------------------------------------------

# %% Prep Prompt

target_words = "aspiring human resources"

results = collection.query(
    query_texts=[target_words],
    n_results=10
)
print(results)

job_tiles_results = results['documents'][0]
list_string = "\n".join(
    f"{i+1} {title}" for i, title in enumerate(job_tiles_results)
    )

prompt_user_1 = f'Sort the relavant job tiles by most to least relavant to the\
target words: {target_words}.'

prompt_user_2 = f'Give a similarity score for each job title on their relavance\
to the target words: {target_words} from an integer of 0 to 100.'

prompt_main = f''' {prompt_user_1}\

Relavant Titles:

{list_string}

Return the Answer:
'''

# %% Model



# load w/e model we want to use

# %% Response

# send and display response from model.
