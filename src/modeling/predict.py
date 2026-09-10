from pathlib import Path
import os
from loguru import logger
import typer
import joblib

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import load_csv, save_csv, LogTransformer
import pandas as pd
import mlflow
import dagshub


app = typer.Typer()


dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    model_path= None
    # -----------------------------------------
):
    if model_path == None:
        model_path= "models:/fault_classifier@champion"
        
    model= mlflow.sklearn.load_model(model_uri=model_path)
    test_data= load_csv(input_path)
    pred= model.predict(test_data)
    concat_df= pd.concat([test_data, prediction_df])
    
    logger.info(f"Performing inference for champion model and save it to {PROCESSED_DATA_DIR}.prediction.csv")
    save_csv(PROCESSED_DATA_DIR,concat_df,"prediction")




if __name__ == "__main__":
    app()
