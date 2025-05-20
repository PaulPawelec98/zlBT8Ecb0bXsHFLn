from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import subprocess

from apzivaproject3.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    MODELING_DIR
    )

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing predictions for models...")

    proc = subprocess.run(
        ["python", MODELING_DIR / r"create_predictions.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.success("model predictions complete.")

    logger.info("Performing predictions for ranknet...")

    proc = subprocess.run(
        ["python", MODELING_DIR / r"predict_ranknet_GPU.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.success("ranknet predictions complete.")

    # logger.info("Performing predictions for Flan...")

    # proc = subprocess.run(
    #     ["python", MODELING_DIR / r"predict_Flan_GPU.py"],
    #     capture_output=True,
    #     text=True,
    #     shell=True
    #     )

    # print(proc.stdout)
    # print(proc.stderr)
    # logger.success("ranknet predictions complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
