# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:20:33 2025

@author: Paul
"""

# %% Setup

# data
import os
import sys
from pathlib import Path
import pandas as pd

# models
import torch
import gc

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
    )

# from transformers.utils import default_cache_path

# Setup -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJ_ROOT)

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

try:
    from apzivaproject3.config import (
        PROCESSED_DATA_DIR,
        CACHE_DIR
        )

    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ['HF_HOME'] = str(CACHE_DIR)

except Exception as e:
    print(f"Error importing config: {e}")  # Or handle it in some other way
# -----------------------------------------------------------------------------

# %% Data

df = pd.read_csv(PROCESSED_DATA_DIR / "clean_df.csv")

# Example Prompt

target_string = 'aspiring human resources'

job_titles = df['job_title_string'][:100].tolist()

list_string = "\n".join(
    f"{i+1}. {title}" for i, title in enumerate(job_titles)
    )

prompt = f"""Rank these job titles in order of highest relavance to the target\
 words: {target_string}. Make sure not to duplicate any job titles when\
 returning the list. At the end of each item make sure to include it's\
 original position number from the inital list.'

Here are the job titles to rank:

{list_string}

"""

# %% Setup Device

device = torch.device("cuda")

# %% Load Model TinyLlama

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

# # quant_config = BitsAndBytesConfig(
# #     load_in_4bit=True,               # or `load_in_8bit=True`
# #     bnb_4bit_compute_dtype="float16",
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_quant_type="nf4",
# # )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=CACHE_DIR,
#     # quantization_config=quant_config,
#     device_map="auto"
# )

# %% Load Model Llama3

# model_name = "meta-llama/Meta-Llama-3-8B"

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token

# # quant_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_use_double_quant=True,
# #     # bnb_4bit_compute_dtype=torch.uint8,
# #     llm_int8_enable_fp32_cpu_offload=True,
# #     bnb_4bit_quant_storage=torch.float16
# # )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=CACHE_DIR,
#     quantization_config=quant_config,
#     device_map="auto",
#     torch_dtype="auto"
# )

'''
Seems to infinitely recurse and then either times/maxes out tokens? Both
Llama3 and tinyllama do this...
'''

# %% Llama 3.2 3B

model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     # bnb_4bit_compute_dtype=torch.uint8,
#     llm_int8_enable_fp32_cpu_offload=True,
#     bnb_4bit_quant_storage=torch.float16
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    # quantization_config=quant_config,
    device_map="auto",
    torch_dtype="auto"
)

# %% Llama Prompt

# Tokenize the input text and set attention mask
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True
    )

# Generate predictions (output)
# inputs = {k: v.to(model.device) for k, v in inputs.items()}

max_minutes=10

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_time=60*max_minutes,
        max_new_tokens=25,
        # max_length=100,
        num_return_sequences=1,
        temperature=0.6,
        top_k=100,
        top_p = 0.2,
        do_sample=True,
    )

# Decode the output tokens into a readable string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)

# %% Qwen 2.5

# from transformers import AutoProcessor, AutoModel

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
# model = AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-7B")

'''
I need a different version of transformers, however, if i update this library
then other parts of my project will break...
'''

# %% Qwen2 - 7b - Instruct

# model_name = "Qwen/Qwen2-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Set False for 8-bit
#     bnb_4bit_compute_dtype=torch.float16,
#     llm_int8_enable_fp32_cpu_offload=True
# )

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=CACHE_DIR,
#     quantization_config=quant_config,
#     device_map="auto",
#     trust_remote_code=True
#     )

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


'''
Models that are around 3billion parameters can be loaded entirely on my GPU.
'''

# %% Qwen Prompt

# prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are an expert human resources assistant that excels in ranking candidates."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    temperature=0.2,
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


# %% Deepseek-R1-Distill-Llama-8B

# %% Run Prompt

