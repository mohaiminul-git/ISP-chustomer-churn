from typing import Annotated, List
from pydantic import BaseModel, Field


class Customer_schema(BaseModel):
    is_tv_subscriber: bool
    is_movie_package_subscriber: bool
    subscription_age: Annotated[int, Field(ge=0)]
    bill_avg: Annotated[int, Field(ge=0)]
    remaining_contract: float
    service_failure_count: Annotated[int, Field(ge=0)]
    download_avg: Annotated[float, Field(ge=0)]
    upload_avg: Annotated[float, Field(ge=0)]
    download_over_limit: Annotated[int, Field(ge=0)]    
    
