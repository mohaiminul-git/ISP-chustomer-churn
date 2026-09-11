from fastapi import FastAPI,Request
from contextlib import asynccontextmanager
from src.utils import load_champion_model
import os
import dagshub


dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.model= load_champion_model()
    yield
    app.state.model= None
    
    
def get_model(request: Request):
    return request.app.state.model

    
