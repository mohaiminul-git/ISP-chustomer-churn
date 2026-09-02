import pandas as pd
from loguru import logger
from pathlib import Path
from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR
import typer

app = typer.Typer()


positive_cols= [
       'subscription_age', 'bill_avg', 'remaining_contract',
       'service_failure_count', 'download_avg', 'upload_avg',
       'download_over_limit'
       ]


@app.command()
def clean_data(input_path: Path = RAW_DATA_DIR / "dataset.csv", output_path: Path = INTERIM_DATA_DIR / "cleaned_dataset.csv") -> None:
    if not input_path.is_file():
        logger.error(f"{input_path.name} doesn't exist.")
        raise typer.Exit(code=1)
    
    data= pd.read_csv(input_path)
    cleaned_data= data_cleaning(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to {output_path}")






def data_cleaning(df:pd.DataFrame)-> pd.DataFrame:
    logger.info("Starting data cleaning process...")
    data= df.drop(["id"], axis=1)
    duplicates = data[data.duplicated()]
    data.drop_duplicates(keep='first', inplace=True)
    logger.info(f"Removed {len(duplicates)} duplicate rows.")

    data=data.rename(columns={"reamining_contract": "remaining_contract"})
    #filling missing values with median
    # negetive values with 0
    
    data["missing_remaining_contract"]= data["remaining_contract"].isna().astype(int)
    for col in positive_cols:
        data[col]= data[col].fillna(data[col].median())
        data.loc[data[col]<0, col]= 0
    logger.info("Data cleaning process completed.")
    return data

if __name__ == "__main__":
    app()
    