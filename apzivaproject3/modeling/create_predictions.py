# %% Packages

# data
import pandas as pd
import numpy as np
import pickle

# ml
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
# from mittens import GloVe
import fasttext
import torch


# Setup ---------------------------------------------------------------
import os
import json
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        MODELING_DIR,
        SETTINGS_FILE
        )

    os.chdir(MODELING_DIR)
    import myRankNet  # My ranknet model

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Setup

# Load Data -------------------------------------------------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")
co_occurance = pd.read_csv(PROCESSED_DATA_DIR / "co_occurance_matrix.csv")
co_occurance.set_index(co_occurance.columns[0], inplace=True)
# co_occurance_array = co_occurance.to_numpy()
# -----------------------------------------------------------------------------

# split string ----------------------------------------------------------------
df['job_title_list'] = [
    sublist.split(" ") for sublist in df['job_title_string']
    ]
# -----------------------------------------------------------------------------

# targets ---------------------------------------------------------------------
# variables -------------------------------------------------------------------
with open(PROJ_ROOT / SETTINGS_FILE, 'r') as f:
    settings = json.load(f)

target_string = settings['target_settings']['target_string']
penalty_string = settings['target_settings']['penalty_string']
# -----------------------------------------------------------------------------

target_list = target_string.split(" ")
# -----------------------------------------------------------------------------

# %% Functions


def calculate_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def calculate_cosine_similarity(vec1, vec2):
    result = (
        np.dot(vec1, vec2) /
        (np.linalg.norm(vec1) *
         np.linalg.norm(vec2))
        )
    return result


def cosine_sim_penalty(vec1, vec2, vec3, method=1, alpha=1):
    # sim scores
    sim_1_3 = calculate_distance(vec1, vec3)
    sim_2_3 = calculate_distance(vec2, vec3)

    if method == 1:  # subtraction
        result = sim_1_3 - sim_2_3

    elif method == 2:  # weighted
        result = sim_1_3 * (1 - alpha * sim_2_3)

    elif method == 3:  # modulated
        mod = vec1 - ((1 - alpha) * vec2)
        result = calculate_cosine_similarity(mod, vec3)

    return result


def create_sudo_fitness(x):
    x = x + abs(x.min())
    x = x / abs(x.max())
    return x


# # %% Fuzzy

# fuzzy -----------------------------------------------------------------------
df['fuzzy'] = [
    fuzz.ratio(target_string, text)/100
    for text
    in df['job_title_string']
    ]
# -----------------------------------------------------------------------------

# %% tf idf

# tf idf ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()


def return_tfidf_score(target, title):
    temp = title.copy()  # To avoid appending to actual title in df...
    temp.append(target)
    tfidf = vectorizer.fit_transform(temp)
    sim = cosine_similarity(tfidf[-1], tfidf[:-1])
    return sim.mean()


df['tfidf'] = list(
    map(
        lambda title: return_tfidf_score(
            target_string, title), df['job_title_list']
        )
    )
# -----------------------------------------------------------------------------
'''
Each title has the appended target word onto, then we vectorize, fit, etc... to
then compare to each title word.
'''


# %% word2vec

# word2vec --------------------------------------------------------------------

model = Word2Vec(
    df['job_title_list'],
    vector_size=50,
    window=3,
    min_count=1,
    epochs=5,
    workers=-1
    )

# Create Vector Average for Target
vector_average_target = []

for i in target_list:
    vector_average_target.append(model.wv[i])

vector_average_target = np.mean(vector_average_target, axis=0)


def return_word2vec_score(model, target_vector, title, func):

    vector_average_title = []

    for i in title:
        vector_average_title.append(model.wv[i])

    vector_average_title = np.mean(vector_average_title, axis=0)

    try:
        result = func(target_vector, vector_average_title)

    except Exception as e:
        print(f"Error, word2vec: {e}")
        result = 0

    return result


df['word2vec'] = list(
    map(
        lambda x: return_word2vec_score(
            model,
            vector_average_target,
            x,
            calculate_cosine_similarity),
        df['job_title_list']
        )
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

# %% GloVe

# GloVe -----------------------------------------------------------------------
idx = {col: idx for idx, col in enumerate(co_occurance.columns)}

# load model
with open(MODELS_DIR / "myglove/my_glove_model.pkl", "rb") as file:
    glove = pickle.load(file)

# Create Vector Average for Target Words.
vector_average_target = []

for i in target_list:
    vector_average_target.append(glove[idx[i]])

vector_average_target = np.mean(vector_average_target, axis=0)


# predict
def return_glove_score(model, target_vector, title, func):

    vector_average_title = []
    for i in title:  # per target, check each word in title for score
        try:

            vector_average_title.append(glove[idx[i]])
        except Exception as e:
            # Get the length of the last vector in the list
            last_vector_length = len(vector_average_title[-1])

            # Create a zero vector of the same length
            zero_vector = [0] * last_vector_length
            vector_average_title.append(zero_vector)
            print(f"Error1, glove {e}")

    vector_average_title = np.mean(vector_average_title, axis=0)

    try:
        result = func(target_vector, vector_average_title)

    except Exception as e:
        print(f"Error2, glove: {e}")
        result = 0

    return result


df['GloVe'] = list(
    map(
        lambda x: return_glove_score(
            glove,
            vector_average_target,
            x,
            calculate_cosine_similarity
            ),
        df['job_title_list']
        )
    )

# similarity = cosine_similarity(vec1, vec2)
# -----------------------------------------------------------------------------
'''
Interesting results, when sorted from least to greatest, we see a bunch of more
hr related job titles appear at the top while near the bottom it's completely
overtaken by data analyst related titles.
'''


# %% fasttext

# fasttext --------------------------------------------------------------------
fsttxt = fasttext.load_model(
    str(MODELS_DIR / 'myfasttext/my_fasttext_model.bin')
    )

# Create Vector Average for Target Words.
vector_average_target = []

for i in target_list:
    vector_average_target.append(fsttxt[i])

vector_average_target = np.mean(vector_average_target, axis=0)


# predict
def return_fasttext_score(model, target_vector, title, func):

    vector_average_title = []
    for i in title:  # per target, check each word in title for score
        try:
            vector_average_title.append(fsttxt[i])
        except Exception as e:
            # Get the length of the last vector in the list
            last_vector_length = len(vector_average_title[-1])

            # Create a zero vector of the same length
            zero_vector = [0] * last_vector_length
            vector_average_title.append(zero_vector)
            print(f"Error1, fasttext {e}")

    vector_average_title = np.mean(vector_average_title, axis=0)

    try:
        result = func(target_vector, vector_average_title)

    except Exception as e:
        print(f"Error2, fasttext: {e}")
        result = 0

    return result


df['fasttext'] = list(
    map(
        lambda x: return_fasttext_score(
            fasttext,
            vector_average_target,
            x,
            calculate_cosine_similarity
            ),
        df['job_title_list']
        )
    )
# -----------------------------------------------------------------------------

# %% BERT

# bert ------------------------------------------------------------------------

# load embedding
embeds = torch.load(
    str(MODELS_DIR / "mybert/mybertembeddings.pt"), weights_only=False
    )

# With Pooler
# pooler = embeds.pooler_output

# df['bert'] = [
#     calculate_cosine_similarity(pooler[-1], title)
#     for title
#     in pooler[:-1]
#     ]

# With hidden_state
CLSs = embeds.last_hidden_state[:, 0, :]  # CLS is entire sentence embedding.

# axis 0: batch_size, so the number of sentences passed (1255)

# axis 1: different tokens in the sequence, CLS, the token that represents the
# whole sentence is the first token.

# axis 2: hidden_state, the dimension of each token (768) This is from the
# design of the BERT model.

df['bert'] = [
    calculate_cosine_similarity(CLSs[-1], title)
    for title
    in CLSs[:-1]
    ]

average_token_embed = embeds.last_hidden_state[:, 1:, :].mean(dim=1)

df['bert2'] = [
    calculate_cosine_similarity(average_token_embed[-1], title)
    for title
    in average_token_embed[:-1]
    ]

# -----------------------------------------------------------------------------
'''
Compare target sentence to each title sentence.

 - bert: This is the CLS of each sentence compared against the target. The
 results were fairly poor.

 - bert2: This is all tokens from the sequence averaged out excluding the
 initial CLS token. Results are much better.

'''

# bert seperate ---------------------------------------------------------------
# embeds1 = torch.load(str(MODELS_DIR / "mybert/mybertembeddings1.pt"))
# embeds2 = torch.load(str(MODELS_DIR / "mybert/mybertembeddings2.pt"))

# CLSs1 = embeds1.last_hidden_state[:, 0, :]
# CLSs2 = embeds2.last_hidden_state[:, 0, :]

# df['bert2'] = [
#     calculate_cosine_similarity(CLSs2, title)
#     for title
#     in CLSs1
#     ]
# -----------------------------------------------------------------------------

'''
No difference between appending or seperating these BERT embeddings.
'''

# %% SBERT

# sbert -----------------------------------------------------------------------

# load sbert embeddings
embeds = torch.load(
    str(MODELS_DIR / "mybert/mysbertembeddings.pt"), weights_only=False
    )

# find similarity
df['sbert'] = [
    calculate_cosine_similarity(embeds[-1], title)
    for title
    in embeds[:-1]
    ]
# -----------------------------------------------------------------------------

'''
sbert worked much better than bert.
'''

# %% SBERT w/ penalty

# sbert -----------------------------------------------------------------------

# load sbert embeddings
embeds = torch.load(
    str(MODELS_DIR / "mybert/mysbertembeddingswpenalty.pt"), weights_only=False
    )

# find similarity
df['sbertwpenalty1'] = [
    cosine_sim_penalty(embeds[-2], embeds[-1], title)
    for title
    in embeds[:-2]
    ]

df['sbertwpenalty2'] = [
    cosine_sim_penalty(embeds[-2], embeds[-1], title, method=2, alpha=0.5)
    for title
    in embeds[:-2]
    ]

df['sbertwpenalty3'] = [
    cosine_sim_penalty(embeds[-2], embeds[-1], title, method=3, alpha=0.5)
    for title
    in embeds[:-2]
    ]
# -----------------------------------------------------------------------------

'''
Definetly improved the results of the search. Most of the top ones are much
closer to the actual target string given.

There isn't a huge difference between each penalty.'
'''

# %% RankNet

# # rankdata
# rankdata = pd.read_csv(PROCESSED_DATA_DIR / "rankdata.csv")

# # variables...
# input_size = 7  # from training
# hidden_size = 32

# # Recreate the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = myRankNet.RankNet(input_size, hidden_size)
# model.load_state_dict(torch.load(
#     MODELS_DIR / "myranknet/myranknet_2025-04-10_16-58-13.pt")
#     )
# model.to(device)

# # Get Model Results
# model.eval()

# # Load and Setup Data
# X = rankdata.loc[:, rankdata.columns[:-2]]

# # Add In Starred Candidates...
# with open(SETTINGS_FILE, 'r') as f:
#     settings = json.load(f)

# star_vec = np.zeros(len(X), dtype=int) + 1
# star_vec[settings['starred_candidates']] = 0

# X['star'] = star_vec
# X = X.values

# X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# y = rankdata['rank']

# scores = {index: 0 for index in df.index}

# with torch.no_grad():
#     for i in range(len(X_tensor)):

#         xi = X_tensor[i].unsqueeze(0)

#         for j in range(len(X_tensor)):

#             if i == j or j > i:
#                 continue

#             xj = X_tensor[j].unsqueeze(0)

#             diff = model(xi, xj)

#             if diff.item() > 0:
#                 scores[i] += 1
#             else:
#                 scores[j] += 1

#             print(i, j)

# # Add scores to dataframe and rank
# rankdata["score"] = scores.values()
# rankdata['job_title_string'] = df['job_title_string']

# print(rankdata[["score", "rank"]].head())

# # parameters...
# for name, param in model.named_parameters():
#     print(f"{name}:\n{param.data}\n")

# rankdata.to_csv(PROCESSED_DATA_DIR / 'df_with_rank.csv', index=False)

# %% Export Results

ignore = ['job_title_string', 'country', 'job_title_list']

df_sudo = df.copy()  # Just fits the score to be 0 - 1.

df_sudo[df.columns.difference(ignore)] = (
    df_sudo[df.columns.difference(ignore)].apply(create_sudo_fitness)
    )

df.to_csv(PROCESSED_DATA_DIR / 'df_with_scores.csv', index=False)
df_sudo.to_csv(PROCESSED_DATA_DIR / 'df_with_fit.csv', index=False)
