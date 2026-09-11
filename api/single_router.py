
from fastapi import APIRouter,Depends
from src.features import generate_features
from src.modeling.predict import predict_single
import pandas as pd
from api.schema import Customer_schema
from api.lifespan import get_model

router= APIRouter()

@router.post("/single_prediction")
async def single_predict_result(
    request: Customer_schema,
    model= Depends(get_model)
):
    dictionary= request.model_dump()
    df= pd.DataFrame([dictionary])
    final_df= generate_features(df)
    pred= predict_single(final_df,model)
    return pred

