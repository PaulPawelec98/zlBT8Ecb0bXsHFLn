# -*- coding: utf-8 -*-
"""
Created on Sun May  4 21:39:58 2025

@author: Paul
"""

# %% Packages

# data
import os
import sys
# import ast
from pathlib import Path
import pandas as pd
import random
import re
from datetime import datetime
from datasets import Dataset
import json

# models
import torch

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
    )

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    get_peft_model,
    TaskType
    )

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling
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
        FINETUNING_DIR
        )

    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ['HF_HOME'] = str(CACHE_DIR)

    if str(CLASSES_DIR) not in sys.path:
        sys.path.append(str(CLASSES_DIR))

    import WindowSlider as ws  # Custom Class for Sliding

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Varibales

model_output = FINETUNING_DIR / 'qwen3'


# %% Load Data

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

# %% Training Data

df_complete = pd.DataFrame()
df_complete['X'] = df['job_title_string']
df_complete['y'] = df['average_sim_score']

# Shuffle and split the data into training (80%), eval (10%), and test (10%)
train_df, temp_df = train_test_split(df_complete, test_size=0.2, random_state=42)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # Splitting remaining 20% into eval and test

# Convert to required format (input, output) for training, eval, and test
def convert_to_required_format(df):
    return [{"text": f"{row['X']} {int(row['y'])}"} for _, row in df.iterrows()]

train_data = convert_to_required_format(train_df)
eval_data = convert_to_required_format(eval_df)
test_data = convert_to_required_format(test_df)

# Save the formatted data as JSON files
with open('train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('eval_data.json', 'w') as f:
    json.dump(eval_data, f, indent=4)

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)

# Optional: Print out the first few entries of each dataset for inspection
print(f"Train Data Sample: {train_data[:3]}")
print(f"Eval Data Sample: {eval_data[:3]}")
print(f"Test Data Sample: {test_data[:3]}")

# %% Model

model_name = 'Qwen/Qwen3-4B-Base'
device = "cuda"

# Quantization configuration

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=512,
    use_fast=True,
    cache_dir=CACHE_DIR,
    padding_side="left"
    )

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    )

model.config.pretraining_tp = 1

# %% Lora Config

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

model = get_peft_model(model, peft_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# %%  Training Settings

# Settings
project = "qwen3-finetune-simscores"
base_model_name = "Qwen3-4B-Base"
run_name = base_model_name + "-" + project
output_dir = str(model_output / 'output' / run_name)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    num_train_epochs=5,
    logging_steps=10,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-4,  # 2e-4, 1e-4, 5e-5
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none"
)

# Convert list to Dataset
train_dataset = Dataset.from_list(train_data)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
)

# %% Train Model

import gc, torch
gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()

# %% Save Model

new_model_name = "Qwen-3-4B-HumanResourcesSimScores"
model.save_pretrained(model_output / new_model_name)
tokenizer.save_pretrained(model_output / new_model_name)

# %% Test Model

target_string = 'aspiring human resources'

# slt = [122, 206, 209, 3, 2, 8, 124, 12, 7, 9, 5, 10]  # specific list
# job_titles = df['job_title_string'][slt].tolist()

job_titles = df['job_title_string'][:20].tolist()
job_scores = df['average_sim_score'][:20].tolist()

list_string = "\n".join(
    f"{i+1}. {title}" for i, title in enumerate(job_titles)
    )

# Message 1 -------------------------------------------------------------------
system_role = "You are an assistant that helps evaluate and sort job titles by\
 their relevance to a target phrase."

example_lst = list(range(0, len(job_titles)))
random.shuffle(example_lst)

prompt = f"""
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

# prompt = "Give me a short introduction to large language model."
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

# max_minutes=10

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs["attention_mask"],
    max_new_tokens=100,
    temperature=0.5,
    # max_time=60*max_minutes,
    # repetition_penalty=1.1,
    do_sample=True,
    use_cache=True,
)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids
    in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


# %% Validate

def clean_output(string):
    result = ' '.join(re.findall(r'\d+', string))
    return result

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
