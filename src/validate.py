from pathlib import Path
from loguru import logger
import pandas as pd
import typer
from src.config import RAW_DATA_DIR

from pydantic import  TypeAdapter, ValidationError 
from src.utils import DataScheme

app= typer.Typer()


@app.command()
def validate_schema_and_properties(dataset_path: Path = RAW_DATA_DIR / "dataset.csv") -> None:
    schema= False
    if not dataset_path.is_file():
        logger.error(f"{dataset_path.name} doesn't exist.")
        raise typer.Exit(code=1)
    
    data= pd.read_csv(dataset_path)
    try:
        TypeAdapter(list[DataScheme]).validate_python(data.to_dict(orient="records"))
        logger.success("Dataset schema validation passed.")
        schema= True
        if schema:
            validate_data(data)        
    except ValidationError as e:
        logger.error(f"Dataset schema validation failed: {e}")
        raise typer.Exit(code=1)
    
    



Expected_columns= {'id', 'is_tv_subscriber', 'is_movie_package_subscriber',
       'subscription_age', 'bill_avg', 'reamining_contract',
       'service_failure_count', 'download_avg', 'upload_avg',
       'download_over_limit', 'churn'}



def validate_data(df: pd.DataFrame) -> list[str]:
    missing_cols= Expected_columns - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing expectecd columns : {missing_cols}")
    if not df["churn"].dropna().isin([0,1]).all():
        raise ValueError("Target variable (Churn) contains values other than 0/1 ")
    
    positive_cols= ['is_tv_subscriber', 'is_movie_package_subscriber',
       'subscription_age', 'bill_avg', 'reamining_contract',
       'service_failure_count', 'download_avg', 'upload_avg',
       'download_over_limit']
    
    warning=[]
    for col in positive_cols:
        if (df[col].dropna()<0).any():
            negetive_values= df.loc[df[col].dropna()<0,["id",col]]
            if len(negetive_values)>0:
                for _, row in negetive_values.iterrows():   
                    id= row["id"]
                    value= row[col]
                    message= f"{col} with id {id} has values smaller than 0 with value {value}"
                    logger.warning(message)
                    warning.append(message)
                

    if warning:
        logger.warning("Dataset validation completed with {} warning(s).",
        len(warning))              
    else:
        logger.success("validation of dataset properties passed")
    
    return warning
        
    
        
if __name__ == "__main__":
    app()
    