# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 22:54:39 2025

@author: Paul
"""

# %% Setup

# data
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
        PROCESSED_DATA_DIR
        )

    '''
    When using subprocess, the path for FUNCTIONS_DIR, can't be found?
    I can add manually for now.
    '''

    os.chdir(PROJ_ROOT)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% ranked data

# creating ranked data --------------------------------------------------------
# path for final scores from predictions script
scores_path = PROCESSED_DATA_DIR / 'df_with_scores.csv'

# features
cols_to_keep = [
     'tfidf',
     'word2vec',
     'GloVe',
     'fasttext',
     'bert2',
     'sbert',
     ]

if os.path.exists(scores_path):
    rankdata = pd.read_csv(scores_path)
    rankdata = rankdata.loc[:, cols_to_keep]
    rankdata['average_score'] = rankdata.mean(axis=1)
    rankdata = rankdata.sort_values(by='average_score', ascending=False)
    rankdata['rank'] = range(len(rankdata), 0, -1)
    rankdata = rankdata.sort_index(ascending=True)
# -----------------------------------------------------------------------------
'''
Drop penalty rankings because they are not in -1 to 1.
take average score, sort descending, and then use this to give a ranking.
'''

rankdata.to_csv(PROCESSED_DATA_DIR / 'rankdata.csv', index=False)
