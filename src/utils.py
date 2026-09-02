
from pydantic import BaseModel, Field, ConfigDict



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
    
