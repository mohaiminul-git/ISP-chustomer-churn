from pathlib import Path
import pandas as pd

from loguru import logger
import typer
import numpy as np

from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def feature_generation(
    input_path: Path = INTERIM_DATA_DIR/ "cleaned_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
):
    logger.info("Generating features from dataset...")
    data= pd.read_csv(input_path)
    processed_data= generate_features(data)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Processed data is added to the {output_path.parent} directory.")

    

def generate_features(df:pd.DataFrame) -> pd.DataFrame:
    df["num_services_subscribed"]= df[["is_tv_subscriber", "is_movie_package_subscriber"]].sum(axis=1)
    total_usage= df[["download_avg", "upload_avg"]].sum(axis=1)
    df["cost_per_usage"]= np.where(
        total_usage>0,
        df["bill_avg"]/total_usage,
        0
    )
    logger.success("Feature engineering complete")
    return df



if __name__ == "__main__":
    app()
