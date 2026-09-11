from pathlib import Path
import os
from loguru import logger
import typer
import pandas as pd
from src.config import  PROCESSED_DATA_DIR
from src.utils import load_csv, save_csv
import mlflow
import dagshub
from typing import Annotated, Literal
from pydantic import Field


app = typer.Typer()


dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "X_test.csv",
    prediction_type: Annotated[
        Literal["single", "Batch"],
        typer.Option()
        ]="single",
    sklearn_model_path= None
    ):
    
    test_data= load_csv(input_path)
    if sklearn_model_path == None:
        sklearn_model_path= "models:/fault_classifier@champion"
    
    model= mlflow.sklearn.load_model(sklearn_model_path)

    if prediction_type=="Batch":
        pred= batch_prediction(test_data,model)
        prediction_df= pd.DataFrame(pred["prediction"], columns=["prediction"])
        concat_df= pd.concat([test_data, prediction_df],axis=1)
        save_csv(PROCESSED_DATA_DIR,concat_df,"prediction")
        logger.info(f"Performing inference for champion model and save it to {PROCESSED_DATA_DIR}.prediction.csv")
    else:
        pred= predict_single(test_data,model)
        print(pred)

    logger.info("Inference from the champion model is complete")
            
        
  

def predict_single(input_df: pd.DataFrame, model):
    pred= model.predict(input_df)
    return {"churn_prediction": int(pred[0])}

def batch_prediction(input_df: pd.DataFrame, model):
    pred= model.predict(input_df)
    return {"prediction" : pred.tolist()}
    
    



if __name__ == "__main__":
    app()
