import yaml
from src.config import PROJ_ROOT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from loguru import logger


with open(PROJ_ROOT/"param_config.yaml", "r") as f:
    param_config= yaml.safe_load(f)
    
    
models= {
    "logistic_regression": lambda:LogisticRegression(solver="liblinear"), "random_forest" :lambda : RandomForestClassifier()
}

def estimatore_builder(model_name:str):
    try:
        clf= models[model_name]()
        return clf
    except Exception:
        logger.error("model name is not valid. Available model logistic_regression and random_forest")
        
def model_param_getter(model_name: str):
    try:
        params_dict= param_config["models"][model_name]["params"]
        return params_dict
    except Exception:
        logger.error("model name is not valid. Available model logistic_regression and random_forest")
    
    