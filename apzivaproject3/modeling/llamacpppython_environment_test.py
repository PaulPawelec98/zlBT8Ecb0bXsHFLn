# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:53:49 2025

@author: Paul
"""

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

# %% Qwen 3 8B Quantized

model = Llama.from_pretrained(
    repo_id="unsloth/Qwen3-8B-GGUF",
    filename="Qwen3-8B-UD-Q3_K_XL.gguf",
    cache_dir=CACHE_DIR,
    enable_thinking=False,
    device="auto",
    n_gpu_layers=30,
)

# %% Qwen 3 Prompt Single

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
You are an expert job title relevance scorer. Your task is to evaluate the similarity between a list of job titles and a set of target words, assigning a relevance score between 0 and 100 for each job title.  100 represents perfect relevance, and 0 represents no relevance.

**Key Considerations for Scoring:**

*   **Relevance to Target Words:**  Prioritize job titles that contain the target words or closely related synonyms.
*   **Word Count Penalty:**  Penalize job titles with a high proportion of words unrelated to the target words.  The more unrelated words, the lower the score.  Consider these extra words as "noise."
*   **Meaning and Context:** Consider the overall meaning and context of the job title.  A title might not contain the exact target words but still be highly relevant.
*   **Specificity:** More specific job titles (that directly address the target words) should generally score higher than vague or general titles.
*   **Avoid simple keyword matching:** Don't just count the number of target words present. Focus on the overall relevance to the provided target words.

**Input:**

Target Words: {target_string}

Job Titles:
{list_string}
"""

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
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
        max_new_tokens=50,
        temperature=0.5,
        do_sample=True,
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
        )[0]

    print(list_string)
    print(response)

    try:
        scores = list(map(int, response.strip().split()))
        print(scores)
        print(len(scores))

        if len(scores) < len(job_titles):
            # Pad with None if model returned too few scores
            scores_adjusted = scores + [None] * (len(job_titles) - len(scores))
        elif len(scores) > len(job_titles):
            # Truncate if model returned too many scores
            scores_adjusted = scores[:len(job_titles)]
        else:
            scores_adjusted = scores

        all_scores.extend(scores_adjusted)

    except ValueError:
        print(f"Non-integer values in model output: {response}")
        all_scores.extend([None] * len(job_titles))

df['generated_sim_score'] = all_scores

base_name = 'df_with_llm_sim_scores'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{base_name}_{timestamp}.csv"

df.to_csv(PROCESSED_DATA_DIR / file_name)

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
You are an expert job title relevance scorer. Your task is to evaluate the similarity between a list of job titles and a set of target words, assigning a relevance score between 0 and 100 for each job title.  100 represents perfect relevance, and 0 represents no relevance.

**Key Considerations for Scoring:**

*   **Relevance to Target Words:**  Prioritize job titles that contain the target words or closely related synonyms.
*   **Word Count Penalty:**  Penalize job titles with a high proportion of words unrelated to the target words.  The more unrelated words, the lower the score.  Consider these extra words as "noise."
*   **Meaning and Context:** Consider the overall meaning and context of the job title.  A title might not contain the exact target words but still be highly relevant.
*   **Specificity:** More specific job titles (that directly address the target words) should generally score higher than vague or general titles.
*   **Avoid simple keyword matching:** Don't just count the number of target words present. Focus on the overall relevance to the provided target words.

**Input:**

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
        scores = re.search(r'\d+$', response).group()
        # scores = list(map(int, response.strip().split()))
        print(scores)
        print(len(scores))

        if len(scores) < len(job_titles):
            # Pad with None if model returned too few scores
            scores_adjusted = scores + [None] * (len(job_titles) - len(scores))
        elif len(scores) > len(job_titles):
            # Truncate if model returned too many scores
            scores_adjusted = scores[:len(job_titles)]
        else:
            scores_adjusted = scores

        all_scores.extend(scores_adjusted)

    except ValueError:
        print(f"Non-integer values in model output: {response}")
        all_scores.extend([None] * len(job_titles))

df['generated_sim_score'] = all_scores

base_name = 'df_with_llm_sim_scores'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{base_name}_{timestamp}.csv"

df.to_csv(PROCESSED_DATA_DIR / file_name)
