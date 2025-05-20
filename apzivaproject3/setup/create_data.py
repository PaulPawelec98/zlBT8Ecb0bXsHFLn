# =============================================================================
# Imports
# =============================================================================


# %% Setup

# data
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter

# data
import us

# Project Setup ---------------------------------------------------------------
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR
        )

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------
'''
Setup so this script can run by itself and with main.
'''

# %% Load

# load data -------------------------------------------------------------------
df1 = pd.read_csv(RAW_DATA_DIR / "potential-talents.csv")
df2 = pd.read_excel(
    RAW_DATA_DIR / "Extended Dataset for Potential Talents.xlsx"
    )

df2 = df2.rename(columns={'title': 'job_title'})
df2 = df2.rename(columns={'location': 'country'})

df = pd.DataFrame()  # blank dataframe to add to
# -----------------------------------------------------------------------------

# %% Clean Job Titles

# clean job_titles ------------------------------------------------------------

# variables
job_titles = pd.concat([df1['job_title'], df2['job_title']])
stop_words = set(stopwords.words('english'))  # stopwords to remove


def clean_words(x):
    x = str(x)
    x = x.strip().lower().replace("  ", "")
    x = re.sub(r'[^a-zA-Z\s]', '', x)  # removes symnbols, nums, etc...
    x = x.strip()
    x = x.replace('\n', ' ')
    x = x.replace(r'  ', ' ')
    x = list(filter(None, x.split(" ")))  # sometimes text came back as ''
    return x


df['job_title_list'] = [
    [word for word in clean_words(word) if word not in stop_words]
    for word
    in job_titles
    ]

df['job_title_string'] = [
    (" ").join([word for word in clean_words(word) if word not in stop_words])
    for word
    in job_titles
    ]

df['job_title_string'] = df['job_title_string'].str.replace(
    r'\s+', ' ', regex=True
    )  # weird double space issue row 44

# -----------------------------------------------------------------------------
'''
Cleaned job title text by using lower, strip, and removed all symbols and numbe
rs

removed stopwords as well.

Also created two entries for df:
 - job_title_list, which splits each word in the job title into a list.
 - job_title_string, which turns the list into a raw string.
'''

# checks ----------------------------------------------------------------------
counts = Counter(
    [word for sublist in df['job_title_list'] for word in sublist]
    )

counts = counts.most_common()
# -----------------------------------------------------------------------------

# %% Clean Locations

# clean locations -------------------------------------------------------------
locations = df1['location']


def clean_locations(x):
    x = x.replace("Area", "").strip()
    x = x.split(',')
    x = [item.strip() for item in x]
    x = list(filter(None, x))
    return x


# clean locatios
locations = [clean_locations(location) for location in locations]

# split into city and state
df1['city'] = [x[0] for x in locations]
df1['state'] = [x[1] if len(x) > 1 else None for x in locations]

# find missing states based on cities
_mask = df1['state'].isna()
city_state = {
    'Greater New York City': 'New York',
    'San Francisco Bay': 'California',
    'Greater Philadelphia': 'Pennsylvania',
    'Greater Boston': 'Massachusetts',
    'Dallas/Fort Worth': 'Texas',
    'Greater Atlanta': 'Georgia',
    'Greater Chicago': 'Illinois',
    'Greater Los Angeles': 'California',
    }

df1.loc[_mask, 'state'] = [
    city_state[city]
    if city in city_state.keys()
    else None
    for city
    in df1.loc[_mask, 'city']
    ]

# Make sure all States are legit
us_states = [state.name for state in us.states.STATES]

df1['state'] = [
    state if state in us_states else None for state in df1['state']
    ]

df1['country'] = [
    'United States' if state in us_states else None for state in df1['state']
    ]

# find countries for not in the states
outliers = df1[df1['country'].isna()]
set(outliers['city'])

city_country = {
    'Amerika Birleşik Devletleri': 'United States',  # Turkish name for USA?
    'Kanada': 'Sweden',
    'İzmir': 'Turkey'
    }

df1['country'] = [
    country
    if country
    else city_country[city]
    for city, country in zip(df1['city'], df1['country'])
    ]
# -----------------------------------------------------------------------------
'''
only df1 needed cleaning
'''

# combine and set county ------------------------------------------------------
countries = pd.concat([df1['country'], df2['country']]).reset_index(drop=True)
df['country'] = countries
# -----------------------------------------------------------------------------
'''
combine both df1, and df2 columns for country onto df.
'''

# %% Drop Rows

# drop empty rows -------------------------------------------------------------

df = df.dropna(subset=['job_title_string'])
df = df.drop_duplicates(subset=['job_title_string', 'country'])
df['job_title_string'] = df['job_title_string'].astype(str)  # sneaky dtypes..
df = df[df['job_title_string'] != "nan"]
df = df[df['job_title_string'] != ""]
df = df.reset_index(drop=True)

# -----------------------------------------------------------------------------
'''
Some Entries are either blank or duplicates. These get dropped.
I end up dropping, job_title_list, because it's annoying to export and load
while keeping the datatype. I can just split when I need to...
'''

# %% Create Corpus

# create corpus ---------------------------------------------------------------
corpus = (" ").join(
    [word for sublist in df['job_title_list'] for word in sublist]
    )
corpus = corpus.replace("\n", "")  # somehow new lines keep coming in??
# -----------------------------------------------------------------------------

# %% Export

# export ----------------------------------------------------------------------
df = df.drop(columns=['job_title_list'])  # annoying to keep this data type.
df.to_csv(PROCESSED_DATA_DIR / "clean_df.csv", index=False)

with open(PROCESSED_DATA_DIR / "corpus.txt", 'w') as file:
    file.write(corpus)
# -----------------------------------------------------------------------------



conflict = [title for title in df['job_title_string'] if 'conflict' in title]
