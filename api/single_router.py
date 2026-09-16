import time
from fastapi import APIRouter,Depends
from src.features import generate_features
from src.modeling.predict import predict_single
import pandas as pd
from api.schema import Customer_schema
from api.lifespan import get_model
from api.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY

router= APIRouter()

@router.post("/single_prediction")
async def single_predict_result(
    request: Customer_schema,
    model= Depends(get_model)
):
    dictionary= request.model_dump()
    df= pd.DataFrame([dictionary])
    final_df= generate_features(df)
    start= time.perf_counter()
    pred= predict_single(final_df,model)
    PREDICTION_LATENCY.labels(endpoint="single").observe(time.perf_counter()-start)
    
    PREDICTION_COUNTER.labels(endpoint="single", predicted_class= str(pred["churn_prediction"])).inc()
    return pred

