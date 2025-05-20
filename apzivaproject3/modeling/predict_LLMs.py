# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:20:33 2025

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

# %% Prompting

target_string = 'aspiring human resources'

# slt = [122, 206, 209, 3, 2, 8, 124, 12, 7, 9, 5, 10]  # specific list
# job_titles = df['job_title_string'][slt].tolist()

job_titles = df['job_title_string'][:25].tolist()
job_scores = df['average_sim_score'][:25].tolist()

list_string = "\n".join(
    f"{i+1}. {title}" for i, title in enumerate(job_titles)
    )

# Message 1 -------------------------------------------------------------------
system_role = "You are an assistant that helps evaluate and sort job titles by\
 their relevance to a target phrase."

example_lst = list(range(0, len(job_titles)))
random.shuffle(example_lst)

prompt = f"""
You are given a list of job titles and a set of target words. Your task to \
 rank the job titles from most to least relevant to the target words.\
 Return the output as a list in the form of index positions,\
 the number of items in this must be equal to {len(job_titles)} and contain\
 only unique numbers, example output:{example_lst}.

**Now complete the task given the following input:**

Target words: {target_string}

Job titles:
{list_string}

Return only the final ranked list of index positions:
"""
# -----------------------------------------------------------------------------

# Message 2 -------------------------------------------------------------------
# Deepseek RN1 seems okay with this...
# prompt2 = f"""
# ***Task***: I have a list of job title strings and I want you to sort them\
#  based on how relevant they are to the phrase {target_string}. Make sure to\
#  include all job titles given, in this case there must be {len(job_titles)}\
#  job titles total.

# Please sort the list into 3 or more tiers of relevance:

#     Highly Relevant – Contains most or all words from the target phrase, \
# {target_string}, or very close variants.

#     Moderately Relevant – Contains some of the words from the target phrase,
# {target_string}.

#     Low Relevance or Not Relevant – Unrelated words when comparing to the \
# target phrase, {target_string}.

# Job titles:
# {list_string}
# """

prompt2 = f"""
Please sort the list into 2 tiers of relavance just like below, do not include\
 duplicate job titles across tiers:

    High Relevance – Contains most or all words from the target phrase, \
{target_string}, or very close variants.

    Low Relevance or Not Relevant – Contains many unrelated words when \
comparing to the target phrase, {target_string}.

Job titles:
{list_string}
"""

prompt3 = f"""
You are given a list of job titles and a set of target words. Your task is to\
 assign a similarity score to each job title based on how relevant it is to\
 the target words. The score should range from 0 to 100, where 100 means\
 highly relevant and 0 means not relevant at all. Return the output as a space-separated list\
 of integers in the same order as the job titles; for example, 85 75 12. The list must have exactly\
 {len(job_titles)} numbers corresponding to the input job titles.

**Now complete the task given the following input:**

Target words: {target_string}

Job titles:
{list_string}

Return only the list of similarity scores:
"""

# -----------------------------------------------------------------------------

# %% Setup Device

device = torch.device("cuda")

# %% Llama 3.2 1B

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

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

# %% Llama Prompt

# Tokenize the input text and set attention mask
inputs = tokenizer(
    prompt3,
    return_tensors="pt",
    padding=True,
    truncation=True
    )

# Some seem to go to cpu? This forces them all to be on cuda.
inputs = {k: v.to(model.device) for k, v in inputs.items()}

max_minutes=10

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        # max_time=60*max_minutes,
        max_new_tokens=1024,
        # temperature=0.5,
        # top_k=100,
        # top_p = 0.2,
    )

# Decode the output tokens into a readable string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

# Print the generated text
print(generated_text)

'''
Cannot grasp the task at hand...
'''
# %% Llama Prompt w/ chat template added.

# messages = [
#     {"role": "system", "content": system_role},
#     {"role": "user", "content": prompt}
# ]

# # Need to Add Actual Chat Template for this Model.
# chat_template_string = "{% if not add_generation_prompt is defined %}{% set \
# add_generation_prompt = false %}{% endif %}{% for message in messages %}\
# {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>'\
# + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n'\
# }}{% endif %}"

# tokenizer.chat_template = chat_template_string
# inputs = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
# )

# outputs = model.generate(
#     **inputs.to(model.device),
#     max_new_tokens=1024,
#     temperature=0.2,
#     do_sample=True,
#     )

# outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])
# print(outputs[0])

'''
Does not work with chat template, I'm guessing there's one specific for Llama 3
The one I grabbed was for Llama 4.
 - This seems like a whole another rabbit hole...
'''

# %% Qwen2.5 3B Instruct

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

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

# %% Qwen 2 Prompt

# prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": system_role},
    {"role": "user", "content": prompt3}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
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

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

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

# %% Qwen 3 Prompt

# prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": system_role},
    {"role": "user", "content": prompt3}
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

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

# %% Qwen 3 8B Model (Quantized)

llm = Llama.from_pretrained(
    repo_id="unsloth/Qwen3-8B-GGUF",
    filename="Qwen3-8B-UD-Q3_K_XL.gguf",
    cache_dir=CACHE_DIR,
    enable_thinking=False,
)

# output = llm(
#     "Q: What is the capital of France?\nA:",
#     max_tokens=10,
#     temperature=0.7,
#     top_p=0.95,
#     echo=False,
# )

# print(output["choices"][0]["text"])

# messages = [
#     {"role": "system", "content": "/no_think " + system_role},
#     {"role": "user", "content": "What's the capital of France?"}
# ]

# %% Qwen 3 8B Prompt (Quantized)

messages = [
    {"role": "system", "content": "/no_think " +system_role},
    {"role": "user", "content": prompt3}
]

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=100,
    temperature=0.7,
    top_p=0.95
)

print(response["choices"][0]["message"]["content"])

# %% Deepseek R1 Distill-Qwen 1.5B

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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

# %% Deepseek Prompt

# prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": system_role},
    {"role": "user", "content": prompt3}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

# max_minutes=10

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs["attention_mask"],
    max_new_tokens=1024,
    temperature=0.2,
    # max_time=60*max_minutes,
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

# %% Validate Prompt One Responses

# Convert Reranking Response to List
# lst = ast.literal_eval(response)
# new_order = pd.Series(job_titles.copy())
# print(new_order[lst])
# print("New List Length: ", len(new_order[lst]))
# print("Old List Length: ", len(job_titles))
# print("Duplicates?: ", len(lst) != len(set(lst)))



# Convert Sim score to list and check
cleaned_response = clean_output(response)
lst = cleaned_response.split(" ")
lst
print(lst)
print("New List Length: ", len(lst))
print("Old List Length: ", len(job_titles))

df_model_response = pd.DataFrame()
df_model_response['job_titles'] = job_titles

if len(df_model_response) > len(lst):
    lst_adjusted = lst + [None] * (len(df_model_response) - len(lst))
elif len(df_model_response) < len(lst):
    lst_adjusted = lst[:len(df_model_response)]
else:
    lst_adjusted = lst

df_model_response['model_scores'] = lst_adjusted
df_model_response['avg_score'] = job_scores
df_model_response

# %% Validata for RN


def extract_scores(text):
    # Match patterns like '1. 95' and extract the second number
    matches = re.findall(r'(\d+)\.\s*(\d+)', text)
    return [int(score) for _, score in matches]

# lst = extract_scores(response)
lst = (response["choices"][0]["message"]["content"]).split(" ")[1:-1]


print("New List Length: ", len(lst))
print("Old List Length: ", len(job_titles))

df_model_response = pd.DataFrame()
df_model_response['job_titles'] = job_titles

if len(df_model_response) > len(lst):
    lst_adjusted = lst + [None] * (len(df_model_response) - len(lst))
elif len(df_model_response) < len(lst):
    lst_adjusted = lst[:len(df_model_response)]
else:
    lst_adjusted = lst

df_model_response['model_scores'] = lst_adjusted
df_model_response['avg_score'] = job_scores
df_model_response