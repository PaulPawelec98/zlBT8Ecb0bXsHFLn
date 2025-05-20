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

import torch

# from llama_cpp import Llama

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
    )

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
        COLLECTION_DATA_DIR,
        CACHE_DIR,
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

system_role_1 = (
    "You analyze job titles for relevance to a set of target words. "
)

system_role_2 = (
    "You analyze job titles for relevance to a set of target words. "
    "Return only one integer from 0 to 100."
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

prompt_main_2 = f''' {prompt_user_2}\

Relavant Titles:

{list_string}

Return only a space seperated list of integer index positions indicating the new order:
'''


# %% Model - Qwen 3 4B

model_name = 'Qwen/Qwen3-4B'

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set False for 8-bit
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    quantization_config=quant_config,
    device_map="auto",
    )

# %% Prompt - Qwen 3 4B

# create device to use cuda.
device = torch.device("cuda")

# prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": system_role_1},
    {"role": "user", "content": prompt_main}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs["attention_mask"],
    max_new_tokens=500,
    temperature=0.5,
    repetition_penalty=1.1,
    do_sample=True,
)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids
    in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
