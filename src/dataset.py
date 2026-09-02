from pathlib import Path

import kagglehub
from loguru import logger
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def download_dataset(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = RAW_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    if not output_path.is_file():
        logger.info(f" {output_path.name} doesn't exist.")
        path = kagglehub.dataset_download("mehmetsabrikunt/internet-service-churn")
        raw_data_path= Path(path) / "internet_service_churn.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(raw_data_path.read_bytes())
        
        logger.info(f"Downloaded {output_path.name} to {output_path.parent.resolve()}")
    else:
        logger.success("Dataset already exists.")
        


if __name__ == "__main__":
    app()
