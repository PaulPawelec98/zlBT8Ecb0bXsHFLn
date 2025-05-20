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

# %% Load Model

# Load Llama Model ------------------------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,               # or `load_in_8bit=True`
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    quantization_config=quant_config,
    device_map="auto"
)
# -----------------------------------------------------------------------------

# %% Run Content

target_string = 'aspiring human resources'


# list_of_prompts = [
#     f"You're a ranking AI. Given the keywords '{target_string}', choose the most relevant job title below.",
#     "1. aspiring human resources professional",
#     "2. people development coordinator ryan",
#     "Respond with only the number of the most relevant option: 1 or 2."
# ]
# prompt = "\n".join(list_of_prompts)

job_titles = [
    "aspiring human resources professional",
    "people development coordinator ryan",
    "advisory board member celal bayar university",
    "student humber college aspiring human resources generalist",
    "hr senior specialist"
]

list_string = "\n".join(f"{i+1}. {title}" for i, title in enumerate(job_titles))

prompt = f"""You are a ranking AI. Rank the following job titles by how relevant they are to the keywords: "{target_string}".

{list_string}

Respond by listing the numbers in order from most relevant to least relevant. For example: "3, 1, 2, 5, 4"
"""

# Tokenize the input text and set attention mask
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True
    )

# Generate predictions (output)
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_length=2500,
        num_return_sequences=1
    )

# Decode the output tokens into a readable string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)

# %% Lama3?

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.uint8,
#     llm_int8_enable_fp32_cpu_offload=True,
#     bnb_4bit_quant_storage=torch.float16
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B",
#     quantization_config=quant_config,
#     device_map="auto",
#     torch_dtype="auto"
# )