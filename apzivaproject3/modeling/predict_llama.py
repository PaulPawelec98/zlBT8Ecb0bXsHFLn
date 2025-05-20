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

# %% Load Model TinyLlama

# Load Llama Model ------------------------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,               # or `load_in_8bit=True`
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    # quantization_config=quant_config,
    device_map="auto"
)
# -----------------------------------------------------------------------------

# %% Load Model Llama3

model_name = "meta-llama/Meta-Llama-3-8B"

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

# %% Run Prompt

target_string = 'aspiring human resources'

job_titles = [
    "data engineer data analyst data pipelines",
    "aspiring human resources professional",
    "people development coordinator ryan",
    "advisory board member celal bayar university",
    "student humber college aspiring human resources generalist",
    "hr senior specialist"
]

list_string = "\n".join(f"{i+1}. {title}" for i, title in enumerate(job_titles))

prompt = f"""Rank these job titles in order of highest relavance to the target words: {target_string}.

{list_string}

"""

# Tokenize the input text and set attention mask
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True
    )

# Generate predictions (output)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_time=10,
        max_new_tokens=25,
        # max_length=100,
        num_return_sequences=1,
        do_sample=True,
    )

# Decode the output tokens into a readable string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
