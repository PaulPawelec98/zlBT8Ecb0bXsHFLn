# -*- coding: utf-8 -*-
"""
Created on Sun May  4 20:47:52 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
import ast
from pathlib import Path
import pandas as pd
import random
import re
from datetime import datetime

# models
import torch
from llama_cpp import Llama

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
    )

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        MODELS_DIR
        )

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
df2 = pd.read_csv(PROCESSED_DATA_DIR / "df_with_scores.csv")

cols_to_keep = [
     'tfidf',
     'word2vec',
     'GloVe',
     'fasttext',
     'bert2',
     'sbert',
     ]

df['average_sim_score'] = df2[cols_to_keep].mean(axis=1)
df['average_sim_score'] = (df['average_sim_score'] * 100).astype(int)

# %% Functions

def clean_output(string):
    result = ' '.join(re.findall(r'\d+', string))
    return result

# %% Setup Device

device = torch.device("cuda")

# %% Qwen 3 4B

# model_name = 'Qwen/Qwen3-4B'

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
# tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Set False for 8-bit
#     bnb_4bit_compute_dtype=torch.float16,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=CACHE_DIR,
#     quantization_config=quant_config,
#     device_map="auto",
#     )

# %% Qwen 3 4B (Fine Tunned)

model_path = MODELS_DIR / 'finetuning\qwen3\Qwen-3-4B-HumanResourcesSimScores'

# Load the tokenizer and model

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quant_config,
    device_map='auto',
    use_safetensors=True,
    )

# %% Qwen 3 8B Quantized

model = Llama.from_pretrained(
    repo_id="unsloth/Qwen3-8B-GGUF",
    filename="Qwen3-8B-UD-Q3_K_XL.gguf",
    cache_dir=CACHE_DIR,
    n_threads=8,
    n_ctx=400,
    verbose=False,
)

# %% Qwen 3 Prompt Batching

# target_string = 'aspiring human resources'
# batch_size = 10

# all_scores = []

# system_role = (
#     "You analyze job titles for relevance to a set of target words. "
#     "Return only a space-separated list of integers from 0 to 100, in the same order as "
#     "the job titles."
# )

# for i in range(0, len(df), batch_size):
#     job_titles = df['job_title_string'][i:i+batch_size].tolist()

#     # Skip batch if it's empty
#     if not job_titles:
#         continue

#     list_string = "\n".join(
#         f"{j+1}. {title}" for j, title in enumerate(job_titles)
#     )

#     prompt = f"""
# You are given a list of job titles and a set of target words. Your task is to
# assign a similarity score to each job title based on how relevant it is to
# the target words. The score should range from 0 to 100, where 100 means
# highly relevant and 0 means not relevant at all. Return the output as a space-separated list
# of integers in the same order as the job titles; for example, 85 75 12. The list must have exactly
# {len(job_titles)} numbers corresponding to the input job titles.

# **Now complete the task given the following input:**

# Target words: {target_string}

# Job titles:
# {list_string}

# Return only the list of similarity scores:
# """

#     messages = [
#         {"role": "system", "content": system_role},
#         {"role": "user", "content": prompt}
#     ]

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False,
#     )

#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         attention_mask=model_inputs["attention_mask"],
#         max_new_tokens=50,
#         temperature=0.5,
#         do_sample=True,
#     )

#     generated_ids = [
#         output_ids[len(input_ids):]
#         for input_ids, output_ids
#         in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(
#         generated_ids, skip_special_tokens=True
#         )[0]

#     print(response)

#     try:
#         scores = list(map(int, response.strip().split()))
#         print(scores)
#         print(len(scores))

#         if len(scores) < len(job_titles):
#             # Pad with None if model returned too few scores
#             scores_adjusted = scores + [None] * (len(job_titles) - len(scores))
#         elif len(scores) > len(job_titles):
#             # Truncate if model returned too many scores
#             scores_adjusted = scores[:len(job_titles)]
#         else:
#             scores_adjusted = scores

#         all_scores.extend(scores_adjusted)

#     except ValueError:
#         print(f"Non-integer values in model output: {response}")
#         all_scores.extend([None] * len(job_titles))

# df['generated_sim_score'] = all_scores

# df.to_csv(PROCESSED_DATA_DIR / 'df_with_llm_sim_scores.csv')

# %% Qwen 3 Prompt Single

# target_string = 'aspiring human resources'

# all_scores = []

# system_role = (
#     "You analyze job titles for relevance to a set of target words. "
#     "Return only one integer from 0 to 100."
# )

# for i in range(0, len(df), 1):
#     job_titles = [df['job_title_string'][i]]

#     # Skip batch if it's empty
#     if not job_titles:
#         continue

#     list_string = "\n".join(
#         f"{title}" for j, title in enumerate(job_titles)
#     )

# #     prompt = f"""
# # You are given a list of job titles and a set of target words. Your task is to\
# #  assign a similarity score to the job title based on how relevant it is to\
# #  the target words. The score should range from 0 to 100, where 100 means\
# #  highly relevant and 0 means not relevant at all. Make sure to give a larger\
# #  penalty to job titles that contain many words not related to the target words.

# # **Now complete the task given the following input:**

# # Target words: {target_string}

# # Job titles:
# # {list_string}

# # Return only the similarity score:
# # """

#     prompt = f"""
# You are an expert job title relevance scorer. Your task is to evaluate the similarity between a list of job titles and a set of target words, assigning a relevance score between 0 and 100 for each job title.  100 represents perfect relevance, and 0 represents no relevance.

# **Key Considerations for Scoring:**

# *   **Relevance to Target Words:**  Prioritize job titles that contain the target words or closely related synonyms.
# *   **Word Count Penalty:**  Penalize job titles with a high proportion of words unrelated to the target words.  The more unrelated words, the lower the score.  Consider these extra words as "noise."
# *   **Meaning and Context:** Consider the overall meaning and context of the job title.  A title might not contain the exact target words but still be highly relevant.
# *   **Specificity:** More specific job titles (that directly address the target words) should generally score higher than vague or general titles.
# *   **Avoid simple keyword matching:** Don't just count the number of target words present. Focus on the overall relevance to the provided target words.

# **Input:**

# Target Words: {target_string}

# Job Titles:
# {list_string}
# """

#     messages = [
#         {"role": "system", "content": system_role},
#         {"role": "user", "content": prompt}
#     ]

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False,
#     )

#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         attention_mask=model_inputs["attention_mask"],
#         max_new_tokens=50,
#         temperature=0.5,
#         do_sample=True,
#     )

#     generated_ids = [
#         output_ids[len(input_ids):]
#         for input_ids, output_ids
#         in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(
#         generated_ids, skip_special_tokens=True
#         )[0]

#     print(list_string)
#     print(response)

#     try:
#         scores = list(map(int, response.strip().split()))
#         print(scores)
#         print(len(scores))

#         if len(scores) < len(job_titles):
#             # Pad with None if model returned too few scores
#             scores_adjusted = scores + [None] * (len(job_titles) - len(scores))
#         elif len(scores) > len(job_titles):
#             # Truncate if model returned too many scores
#             scores_adjusted = scores[:len(job_titles)]
#         else:
#             scores_adjusted = scores

#         all_scores.extend(scores_adjusted)

#     except ValueError:
#         print(f"Non-integer values in model output: {response}")
#         all_scores.extend([None] * len(job_titles))

# df['generated_sim_score'] = all_scores

# base_name = 'df_with_llm_sim_scores'
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# file_name = f"{base_name}_{timestamp}.csv"

# df.to_csv(PROCESSED_DATA_DIR / file_name)

# %% Qwen 3 Prompt Single Llama

target_string = 'aspiring human resources'

all_scores = []

system_role = (
    "You analyze job titles for relevance to a set of target words. "
    "Return only one integer from 0 to 100."
)

for i in range(0, len(df), 1):
    job_titles = [df['job_title_string'][i]]

    # Skip batch if it's empty
    if not job_titles:
        continue

    list_string = "\n".join(
        f"{title}" for j, title in enumerate(job_titles)
    )

#     prompt = f"""
# You are given a list of job titles and a set of target words. Your task is to\
#  assign a similarity score to the job title based on how relevant it is to\
#  the target words. The score should range from 0 to 100, where 100 means\
#  highly relevant and 0 means not relevant at all. Make sure to give a larger\
#  penalty to job titles that contain many words not related to the target words.

# **Now complete the task given the following input:**

# Target words: {target_string}

# Job titles:
# {list_string}

# Return only the similarity score:
# """

    prompt = f"""
You are an expert in evaluating job title relevance. Score each title from 0 (no relevance) to 100 (perfect match) based on its similarity to the target words.

**Scoring Guidelines:**
 Prioritize titles with target words or close synonyms.
 Penalize titles with many unrelated words (noise).
 Consider overall meaning and context, not just word matching.
 More specific titles should score higher.
 Avoid simple keyword counting; assess true relevance.

Target Words: {target_string}

Job Titles:
{list_string}
"""

    messages = [
        {"role": "system", "content": "/no_think " + system_role},
        {"role": "user", "content": prompt}
    ]

    output = model.create_chat_completion(
        messages=messages,
        max_tokens=10,
        temperature=0.7,
        top_p=0.95
    )

    try:
        response = output["choices"][0]["message"]["content"]
        scores = [re.search(r'\d+$', response).group()]
        # scores = list(map(int, response.strip().split()))
        print('Current JT: ', job_titles)
        print('Response: ', scores)
        print('i == ', i)


        if len(scores) < len(job_titles):
            # Pad with None if model returned too few scores
            scores_adjusted = scores + [None] * (len(job_titles) - len(scores))
        elif len(scores) > len(job_titles):
            # Truncate if model returned too many scores
            scores_adjusted = scores[:len(job_titles)]
        else:
            scores_adjusted = scores

        all_scores.extend(scores_adjusted)

    except Exception as e:
        print(f"Error {e}: {response}")
        all_scores.extend([None] * len(job_titles))

int_all_scores = [int(item) if item is not None else None for item in all_scores]

df['generated_sim_score'] = int_all_scores

base_name = 'df_with_llm_sim_scores'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{base_name}_{timestamp}.csv"

df.to_csv(PROCESSED_DATA_DIR / file_name)
