# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 16:20:16 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
from pathlib import Path

import pandas as pd
import random
import re
import pickle

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        )

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

# Options
pd.set_option('display.max_rows', None)

# Other Variables
target_string = 'aspiring human resources'

# load job titles
df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
df2 = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")
df['rank'] = df2['rank'].copy()

# %% Create Examples

examples_dict = {f'example{i}': None for i in range(1, 11)}


def create_prompt(data, key):
    # create job title strings
    job_titles = data['job_title_string']
    list_string = "\n".join(
        f"{i} {title}" for i, title in enumerate(job_titles)
        )

    # Convert key to string and clean
    clean_key = ' '.join(re.findall(r'\d+', str(key)))

    # Create Prompt
    string = f'''\
Target words:
{target_string}

Job titles:
{list_string}

Return only the final ranked list of index positions: {clean_key}
    '''
    return string


def create_example(data, sample_length=10):
    '''
    Create an example based on a sample of our rankdata
    '''

    # Variables
    indexes = list(range(0, len(data)))
    random_indices = random.sample(range(len(indexes)), sample_length)

    # Sample Data
    sample = data.loc[random_indices, ['job_title_string', 'rank']]
    sample = sample.reset_index()

    # Return Rank and True Index Labels
    ranks = sample['rank'].copy()
    ranks = ranks.sort_values(ascending=False)
    y = ranks.index.tolist()  # how it should be sorted...

    # Create Prompt (str)
    prompt = create_prompt(sample, y)
    return {'titles': sample, 'y': y, 'prompt': prompt}


for item in examples_dict.items():
    result = create_example(df, random.randint(10, 25))
    examples_dict[item[0]] = result


# %% Export

with open(PROCESSED_DATA_DIR / 'openai_examples_for_prompts.pkl', 'wb') as f:
    pickle.dump(examples_dict, f)
