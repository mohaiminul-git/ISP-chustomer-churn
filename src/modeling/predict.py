from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import joblib

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import load_file, save_file
from src.modeling.train import LogTransformer
import pandas as pd


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    model_name= "random_forest.pkl",
    model_path= None
    # -----------------------------------------
):
    if model_path == None:
        model_path= MODELS_DIR/model_name
        
    model= joblib.load(model_path)
    data= load_file(features_path)
    pred= model.predict(data)
    prediction_df= pd.DataFrame({
        "predicton": pred
    })
    logger.info(f"Performing inference for model{model_name} and save it to {PROCESSED_DATA_DIR}.prediction.csv")
    save_file(PROCESSED_DATA_DIR, prediction_df, "prediction")





if __name__ == "__main__":
    app()
