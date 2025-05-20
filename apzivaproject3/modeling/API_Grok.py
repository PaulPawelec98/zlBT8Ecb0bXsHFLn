# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 16:40:49 2025

@author: Paul
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:52:57 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import random
import re
from datetime import datetime
import pickle

# openai
from openai import OpenAI

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        )

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

# load job titles
df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

# load examples rankings
pickle_examples = PROCESSED_DATA_DIR / 'openai_examples_for_prompts.pkl'
with open(pickle_examples, 'rb') as f:
    prompt_examples = pickle.load(f)

# paths
DEFAULT_LOG = MODELS_DIR / "openai"

# %% Connect to Client

# Grok Client -----------------------------------------------------------------

# grab env variables
load_dotenv()

# connect
if 'client' not in globals():
    client = OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=os.getenv("XAI_API_KEY"),
    )

# -----------------------------------------------------------------------------

# %% Test Client

try:
    models = client.models.list()
    print("Connection successful. Models retrieved:")
    for model in models.data[:3]:  # show only the first 3 for brevity
        print("-", model.id)
except Exception as e:
    print("Connection failed:", str(e))

# %% Functions


def log_request(msg_in, msg_out, model, log_path=DEFAULT_LOG):

    file_date = datetime.now().strftime("%Y%m%d")
    file_name = DEFAULT_LOG / f"xAI-{model}-{file_date}.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(f'[{timestamp}]\n')
        f.write(f"INPUT:\n{msg_in}\n")
        f.write(f"OUTPUT:\n{msg_out}\n")
        f.write("=" * 60 + "\n")
    print(f'Updated Log: {file_name}')


def clean_output(string):
    result = ' '.join(re.findall(r'\d+', string))
    return result


# %% Prompts

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
prompt1 = (
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
 rank the job titles from most to least relevant to the target words based on\
 semantic similarity or contextual relevance.

Return the result as a Python list of space-separated zero-based index\
 positions, representing the sorted order of job titles. The list must be\
 exactly {len(job_titles)} elements long, and must contain only unique\
 integers from 0 to {len(job_titles) - 1} (inclusive), with no omissions or\
 duplicates.

Examples:

{prompt_examples['example1']['prompt']}

{prompt_examples['example2']['prompt']}

{prompt_examples['example3']['prompt']}

**Now complete the task given the following input:**

Target words:
{target_string}

Job titles:
{list_string}

Return only the final ranked list of index positions:
"""
)
# -----------------------------------------------------------------------------

# final message(s) used for the model.
messages = [
    {"role": "system", "content": system_role},
    {"role": "user", "content": prompt1}
]

# %% Basic Calls

models = list(client.models.list())
model_names = [m.id for m in models]

# %% ChatGPT

generate = False

try:
    if generate:
        model_name = "gpt-4.1-mini"
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        out = completion.choices[0].message.content

        print('Success, Output Below...\n')
        print(out)

        log_request(messages, out, model_name)

    else:
        pass
except Exception as e:
    print('Error: ', e)

# %% Validate

# convert to list
out_clean = clean_output(out)
index_list = list(map(int, out_clean.split()))

# duplicates
check_dup = len(index_list) == len(set(index_list))
print('No Duplicates?: ', check_dup)

# correct length
check_len = len(index_list) == len(job_titles)
print('Correct Length?: ', check_len)

job_titles[index_list[:100]]
