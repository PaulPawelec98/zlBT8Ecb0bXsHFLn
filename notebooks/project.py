# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:41:09 2025

@author: Paul
"""

# Project 3: HR Talent

# %% Setup

# Packages --------------------------------------------------------------------
import os
import pandas as pd
import re
import numpy as np
from collections import Counter

# locations
import us

# plotting
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# sklearn
# from sklearn.metrics.pairwise import cosine_similarity

# ranking
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from mittens import Mittens, GloVe

# -----------------------------------------------------------------------------

# Directory -------------------------------------------------------------------
root = r'E:\My Stuff\Projects\Apziva\868456845468568'
raw_dir = r'E:\My Stuff\Projects\Apziva\868456845468568\data\raw'
os.chdir(raw_dir)

# -----------------------------------------------------------------------------

# data and settings -----------------------------------------------------------
df = pd.read_csv('potential-talents.csv')
seed = 3921
np.random.seed(seed)

pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Adjust width for long outputs
# -----------------------------------------------------------------------------

'''
## Background:

As a talent sourcing and management company, we are interested in finding
talented individuals for sourcing these candidates to technology companies.
Finding talented candidates is not easy, for several reasons. The first reason
is one needs to understand what the role is very well to fill in that spot,
this requires understanding the client’s needs and what they are looking
for in a potential candidate. The second reason is one needs to understand
what makes a candidate shine for the role we are in search for. Third,
where to find talented individuals is another challenge.

The nature of our job requires a lot of human labor and is full of manual
operations. Towards automating this process we want to build a better
approach that could save us time and finally help us spot potential
candidates that could fit the roles we are in search for. Moreover,
going beyond that for a specific role we want to fill in we are interested
in developing a machine learning powered pipeline that could spot talented
individuals, and rank them based on their fitness.

We are right now semi-automatically sourcing a few candidates, therefore the
sourcing part is not a concern at this time but we expect to first determine
best matching candidates based on how fit these candidates are for a given
role. We generally make these searches based on some keywords such as
“full-stack software engineer”, “engineering manager” or “aspiring human
resources” based on the role we are trying to fill in. These keywords might
change, and you can expect that specific keywords will be provided to you.

Assuming that we were able to list and rank fitting candidates, we then employ
a review procedure, as each candidate needs to be reviewed and then determined
how good a fit they are through manual inspection. This procedure is done
manually and at the end of this manual review, we might choose not the first
fitting candidate in the list but maybe the 7th candidate in the list. If that
happens, we are interested in being able to re-rank the previous list based
on this information. This supervisory signal is going to be supplied by
starring the 7th candidate in the list. Starring one candidate actually sets
this candidate as an ideal candidate for the given role. Then, we expect the
list to be re-ranked each time a candidate is starred.

## Data Description:

The data comes from our sourcing efforts. We removed any field that could
directly reveal personal details and gave a unique identifier for each
candidate.

Attributes:
id : unique identifier for candidate (numeric)
job_title : job title for candidate (text)
location : geographical location for candidate (text)
connections: number of connections candidate has, 500+ means over 500 (text)
Output (desired target):
fit - how fit the candidate is for the role? (numeric, probability between 0-1)
Keywords: “Aspiring human resources” or “seeking human resources”

## Goal(s):

Predict how fit the candidate is based on their available information
(variable fit)

Success Metric(s):

Rank candidates based on a fitness score.
Re-rank candidates when a candidate is starred.
Current Challenges:

We are interested in a robust algorithm, tell us how your solution works and
show us how your ranking gets better with each starring action.

How can we filter out candidates which in the first place should not be in this
list?
    - By location? If that matters, you could pre make a list of okay locations
    or map distances from location to work and then filter based on max or min
    distance to work.
    - remove candidates if they are missing any or all of the keywords.
    - remove candidates based on negative keys? So add in a negative score?
    penalties?
        - could work after, we do initial

Can we determine a cut-off point that would work for other roles without losing
high potential candidates?
    - Context? Pick top X, pick above cuttoff of total percentage of candidates
        - have specific requirements...
        - how many interviews can be feasiably done?

Do you have any ideas that we should explore so that we can even automate this
procedure to prevent human bias?
    - remove anything related to a country, university, language, race,
    politics, and maybe academics?
'''


# %% Explore Data

# Quick Look ------------------------------------------------------------------
df.describe()
df.head()
# -----------------------------------------------------------------------------

# Clean Words -----------------------------------------------------------------
words = df['job_title']


def clean_words(x):
    x = x.strip().lower().replace("  ", " ")
    x = re.sub(r'[^a-zA-Z\s]', '', x)
    x = x.split(" ")
    return x


word_list = [word for sublist in map(clean_words, words) for word in sublist]
Counter(word_list)
# -----------------------------------------------------------------------------

# Plot Words ------------------------------------------------------------------
text = " ".join(word_list)

STOPWORDS = STOPWORDS.union({''})

# Generate word cloud
wordcloud = WordCloud(
    stopwords=STOPWORDS,
    width=800,
    height=400,
    background_color="white",
    collocations=False
    ).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Hide axis
plt.show()
# -----------------------------------------------------------------------------
'''
Most common seem to be...
    - professional, human, resources, aspiring, student, university
'''

# Check Locations -------------------------------------------------------------
locations = df['location'].copy()
Counter(locations)


def clean_locations(x):
    x = x.replace("Area", "").strip()
    x = x.split(',')
    x = [item.strip() for item in x]
    return x


# clean locatios
locations = [clean_locations(location) for location in locations]

# split into city and state
df['city'] = [x[0] for x in locations]
df['state'] = [x[1] if len(x) > 1 else None for x in locations]

# find outliers
_mask = df['state'].isna()
outliers = df[_mask]
Counter(outliers['city'])

# find missing states based on cities
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

df.loc[_mask, 'state'] = [
    city_state[city]
    if city in city_state.keys()
    else None
    for city
    in df.loc[_mask, 'city']
    ]

# Make sure all States are legit
us_states = [state.name for state in us.states.STATES]

df['state'] = [state if state in us_states else None for state in df['state']]

# find wrong states
_mask = df['state'].isna()
outliers = df[_mask]
Counter(outliers['city'])
outliers
# -----------------------------------------------------------------------------
'''
Could drop any candidate that dosen't have a state, since they would not
be based in the US.
'''

# See all cities/state counts -------------------------------------------------
Counter(df['state'])
Counter(df['city'])
# -----------------------------------------------------------------------------
'''
map plot?
'''


# Map Location to City/State --------------------------------------------------
# -----------------------------------------------------------------------------
'''
'''


# %% Feature Engineering

# %% Machine Learning

'''
- GloVe, word2vec
- BERT
'''

# fuzzy -----------------------------------------------------------------------
target_string = 'Aspiring human resources'

df['fuzzy'] = [
    fuzz.ratio(target_string, text)/100
    for text
    in df['job_title']
    ]
# -----------------------------------------------------------------------------

# tf idf ----------------------------------------------------------------------
word_list = [clean_words(word) for word in df['job_title']]

vectorizer = TfidfVectorizer()


def return_tfidf_score(target, title):
    title.append(target)
    tfidf = vectorizer.fit_transform(title)
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim.mean()


df['tfidf'] = list(
    map(lambda title: return_tfidf_score(target_string, title), word_list)
    )
# -----------------------------------------------------------------------------
'''
word list is created by
'''

# word2vec --------------------------------------------------------------------

model = Word2Vec(
    word_list, vector_size=50, window=3, min_count=1, epochs=5, workers=-1
    )

target = target_string.lower().split(" ")


def return_word2vec_score(model, target, title):

    vec_score = []
    for i in target:

        vec_average = []

        for j in title:

            vec_average.append(model.wv.similarity(i, j))

        vec_score.append(np.mean(vec_average))

    return np.mean(vec_score)


df['word2vec'] = list(
    map(lambda x: return_word2vec_score(model, target, x), word_list)
    )


# other functions --------
# model.wv.most_similar('english')
# model.save('word2vec.model')
# model = Word2Vec.load('word2vec.model')
# model.train([[]])
# -----------------------------------------------------------------------------
'''
Corpos is created via each job title going into a seperate item onto a list,
then fed into the word2vec model.

Keyword string is split and fed into the model one by one against each word in
each item returning an average of all averages for each word in the keyword
string.
'''

# GloVe -----------------------------------------------------------------------

# cooccurrence_matrix
model = GloVe(n=50, max_iter=1000)  # 50-dimensional embeddings
word_vectors = model.fit(cooccurrence_matrix)

# -----------------------------------------------------------------------------

# '''
# duplicate candidates? These all have the exact same information.

# 	job_title	location	connection	fit	city	state	fuzzy
# 5	Aspiring Human Resources Specialist	Greater New York City Area	1		Greater New York City	New York	0.7457627118644068
# 23	Aspiring Human Resources Specialist	Greater New York City Area	1		Greater New York City	New York	0.7457627118644068
# 35	Aspiring Human Resources Specialist	Greater New York City Area	1		Greater New York City	New York	0.7457627118644068
# 48	Aspiring Human Resources Specialist	Greater New York City Area	1		Greater New York City	New York	0.7457627118644068
# 59	Aspiring Human Resources Specialist	Greater New York City Area	1		Greater New York City	New York	0.7457627118644068
# 2	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# 16	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# 20	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# 32	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# 45	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# 57	Aspiring Human Resources Professional	Raleigh-Durham, North Carolina Area	44		Raleigh-Durham	North Carolina	0.7213114754098361
# '''

# %% Conclusions