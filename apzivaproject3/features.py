from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import subprocess

from apzivaproject3.config import (
    MODELING_DIR,
    SETUP_DIR
    )

app = typer.Typer()


@app.command()
def main(
    # ---- Paths ----
    # input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- Run Scripts ----
    logger.info("Generating features from dataset...")

    proc = subprocess.run(
        ["python", SETUP_DIR / r"create_features.py"],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.success("Features generation complete.")

    # logger.info("Premake Tokens for Job Titles...")

    # proc = subprocess.run(
    #     ["python", MODELING_DIR / r"premake_tokens.py"],
    #     capture_output=True,
    #     text=True,
    #     shell=True
    #     )

    # print(proc.stdout)
    # print(proc.stderr)

    # logger.success("Premake Tokens complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
