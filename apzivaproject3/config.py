from pathlib import Path

import os
import sys
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# DATA
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
COLLECTION_DATA_DIR = DATA_DIR / 'collections'

# MODELS
MODELS_DIR = PROJ_ROOT / 'models'
CACHE_DIR = r'F:\Local-Models\Local-LLM\models\.cache'  # hugging face
FINETUNING_DIR = MODELS_DIR / 'finetuning'

# MODULES
MODULES_DIR = PROJ_ROOT / "apzivaproject3"
SETUP_DIR = MODULES_DIR / "setup"
MODELING_DIR = MODULES_DIR / "modeling"
FUNCTIONS_DIR = MODULES_DIR / "functions"
CLASSES_DIR = MODULES_DIR / 'classes'

# RESOURCES
RESOURCES_DIR = PROJ_ROOT / "res"

# REPORTS
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# SETTINGS FILE
SETTINGS_FILE = "user.json"

os.chdir(PROJ_ROOT)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135

try:
    # if logger.has(0):  # Check if handler with ID 0 exists
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
