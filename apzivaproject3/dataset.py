from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import subprocess

from apzivaproject3.config import RAW_DATA_DIR, SETUP_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "potential-talents.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "clean.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    proc = subprocess.run(
        ["python", SETUP_DIR / r"create_data.py", RAW_DATA_DIR],
        capture_output=True,
        text=True,
        shell=True
        )

    print(proc.stdout)
    print(proc.stderr)

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
