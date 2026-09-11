
from fastapi import APIRouter,status,UploadFile,File, Depends
import pandas as pd
from src.features import generate_features
from src.modeling.predict import  batch_prediction
from api.lifespan import get_model

router= APIRouter()



@router.post("/batch_prediction", status_code=status.HTTP_200_OK)
async def batch_predict_result(
    request : UploadFile = File(description="upload .csv only") ,
    model= Depends(get_model)
    ):
        
    df= pd.read_csv(request.file)
    final_df= generate_features(df)
    pred= batch_prediction(final_df,model)
    return pred
