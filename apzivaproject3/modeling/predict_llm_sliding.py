# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 20:04:57 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
# import ast
from pathlib import Path
import pandas as pd
import random
import re

# models
import torch

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
        CLASSES_DIR,
        )

    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ['HF_HOME'] = str(CACHE_DIR)

    if str(CLASSES_DIR) not in sys.path:
        sys.path.append(str(CLASSES_DIR))

    import WindowSlider as ws  # Custom Class for Sliding

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

# %% Functions

# %% Prompting

system_role = (
 "You are a concise assistant that ranks job titles by relevance. "
 "Respond only with a space-separated list of zero-based index positions. "
 "Do not explain anything."
)

# Job Titles
job_titles = df["job_title_string"]
list_string = "\n".join(
    f"{i} {title}" for i, title in enumerate(job_titles)
    )

# Example Index List
# example_lst = list(range(0, len(job_titles)))
example_lst = list(range(0, 10))
random.shuffle(example_lst)
example_lst = re.sub(r'[\[\],]', '', str(example_lst[:10])).strip()
example_lst

# Target
target_string = "aspiring human resources"

# prompt 1 --------------------------------------------------------------------
prompt = (
    f"""\
You are given a list of job titles and a set of target words. Your task is to\
 rank the job titles from most to least relevant to the target words based on\
 semantic similarity or contextual relevance.

Return the result as a Python list of space-separated zero-based index\
 positions, representing the sorted order of job titles. The list must be\
 exactly {len(job_titles)} elements long, and must contain only unique\
 integers from 0 to {len(job_titles) - 1} (inclusive), with no omissions or\
 duplicates.

Example output: {example_lst}

**Now complete the task given the following input:**

Target words:
{target_string}

Job titles:
{list_string}

Return only the final ranked list of index positions:
"""
)
# -----------------------------------------------------------------------------

# prompt 2 --------------------------------------------------------------------
prompt2 = (
    f"""\
You are given a list of job titles and a set of target words. Your task is to\
 give similarity score with a value from 0 to 100 (inclusive) based on the\
 relavance of the job title to the target word.

Return the result as a Python list of space-separated numbers, representing
 the similarity of job titles to target words. The list must be exactly\
 {len(job_titles)} elements long.

Example output: 0 50 2 99 55 68

**Now complete the task given the following input:**

Target words:
{target_string}

Job titles:
{list_string}

Return only the final score for each job title:
"""
)
# -----------------------------------------------------------------------------

# %% Setup Device

device = torch.device("cuda")

# %% Qwen2.5 3B Instruct

# model_name = "Qwen/Qwen2.5-3B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

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

# %% Qwen 3 4B

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

# %% Sliding

window_size = 10
slide_size = 5
n_rows = len(df)

# Calculate the number of slides
n_slides = ((n_rows - window_size) // slide_size) + 1
slides = list(range(n_slides))

wsldr = ws.WindowSlider(df, window_size, slide_size)


def validate_response(out, check_idx):

    def clean_output(string):
        result = ' '.join(re.findall(r'\d+', string))
        return result

    # convert to list
    print('out: ', out)
    out_clean = clean_output(out)
    index_list = list(map(int, out_clean.split()))

    # # duplicates
    # check_dup = len(index_list) == len(set(index_list))
    # print('No Duplicates?: ', check_dup)

    # # correct length
    # check_len = len(index_list) == len(job_titles)
    # print('Correct Length?: ', check_len)

    # job_titles[[item for item in index_list if item != 20]]

    return index_list


def create_slide_prompt(data, idx):

    job_titles = df.loc[idx, ["job_title_string"]]
    list_string = "\n".join(
        f"{i} {title}" for i, title in enumerate(job_titles)
        )

    prompt = (
        f"""\
You are given a list of job titles and a set of target words. Your task is to\
 rank the job titles from most to least relevant to the target words based on\
 semantic similarity or contextual relevance.

Return the result as a Python list of space-separated zero-based index\
 positions, representing the sorted order of job titles. The list must be\
 exactly {len(job_titles)} elements long, and must contain only unique\
 integers from 0 to {len(job_titles) - 1} (inclusive), with no omissions or\
 duplicates.

Example output: {example_lst}

**Now complete the task given the following input:**

Target words:
{target_string}

Job titles:
{list_string}

Return only the final ranked list of index positions:
    """
    )
    return prompt


count = 0
for slide in slides:
    old_idx = wsldr.get_current_window()

    print('Current Window', old_idx)

    new_prompt = create_slide_prompt(df, old_idx)

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": new_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # max_minutes=10

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=50,
        temperature=1.5,
        # max_time=60*max_minutes,
        # repetition_penalty=1.1,
        # do_sample=True,
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
        )[0]

    clean_response = validate_response(response, old_idx)
    print('Clean Response', clean_response)

    rebased = [x + (slide_size*count) for x in clean_response]
    print('Rebased:', rebased)

    wsldr.swap(rebased)
    wsldr.slide()
    count += 1


df_downslide = wsldr.df.copy()

# %% Backwards

count = n_slides-1
for slide in slides:
    old_idx = wsldr.get_current_window()

    print('Current Window', old_idx)

    new_prompt = create_slide_prompt(df, old_idx)

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": new_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # max_minutes=10

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.5,
        # max_time=60*max_minutes,
        # repetition_penalty=1.1,
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

    clean_response = validate_response(response, old_idx)
    print('Clean Response', clean_response)

    rebased = [x + (slide_size * count) for x in clean_response]
    print('Rebased:', rebased)

    wsldr.swap(rebased)
    wsldr.slide(direction='up')
    count -= 1

df_upslide = wsldr.df.copy()

df_downslide.to_csv(PROCESSED_DATA_DIR / 'df_upside_2.csv')
df_upslide.to_csv(PROCESSED_DATA_DIR / 'df_upside_2.csv')


# %% Load

df_upside_1 = pd.read_csv(PROCESSED_DATA_DIR / 'df_upside_1.csv')
# df_upslide_1 = pd.read_csv(PROCESSED_DATA_DIR / 'df_upside_1.csv')

# df_downslide.to_csv()
# df_upslide.to_csv(PROCESSED_DATA_DIR / 'df_upside_2.csv')
