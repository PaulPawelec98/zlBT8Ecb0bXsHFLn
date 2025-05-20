from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import subprocess

from apzivaproject3.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MODELING_DIR,
    SETUP_DIR
    )
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")

    proc = subprocess.run(
        ["python", MODELING_DIR / r"create_models.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.success("Modeling training complete.")

    logger.info("Update rankdata...")

    proc = subprocess.run(
        ["python", SETUP_DIR / r"create_rankdata.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.info("Finished rankdata.")

    logger.info("Starting RankNet...")

    proc = subprocess.run(
        ["python", MODELING_DIR / r"create_ranknet_GPU.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.info("Finished Rank Net.")


if __name__ == "__main__":
    app()
