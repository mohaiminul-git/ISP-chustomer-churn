
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from pathlib import Path



class DataScheme(BaseModel):
    model_config=ConfigDict(
        extra="forbid"
    )
    id: int
    is_tv_subscriber: int
    is_movie_package_subscriber: int
    subscription_age: float
    bill_avg: int
    reamining_contract: float
    service_failure_count: int
    download_avg: float
    upload_avg: float
    download_over_limit: int
    churn: int= Field(ge=0, le=1)
    
    
    
def save_file(data_path: Path, df:pd.DataFrame, df_name: str):
    file_path= data_path/f"{df_name}.csv"
    file_path.parent.mkdir(exist_ok=True)
    df.to_csv(file_path, index=False)
    
def load_file(data_path: Path)-> pd.DataFrame:
    data= pd.read_csv(data_path)
    return data