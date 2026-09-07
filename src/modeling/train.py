import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, config
from src.utils import save_file

app = typer.Typer()


class LogTransformer(BaseEstimator, TransformerMixin):
    """log1p on the columns it receives. Stateless, so fit() is a no-op."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return np.log1p(X)


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, config: dict, model_name: str):
        self.df = df
        self.config = config
        self.model_name = model_name

    def split_data(self):
        X = self.df.drop(columns=[self.config["target_col"]])
        y = self.df[self.config["target_col"]]
        return train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=y,
        )

    def build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("log", LogTransformer()),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, self.config["numeric_cols"]),
                ("binary", "passthrough", self.config["binary_cols"]),
            ]
        )
        model_spec = self.config["models"][self.model_name]
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("clf", model_spec["estimator"])]
        )

    def train(self):
        data_dir = PROCESSED_DATA_DIR
        X_train, X_test, y_train, y_test = self.split_data()
        datas = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        for key, value in datas.items():
            save_file(data_dir, value, key)

        pipeline = self.build_pipeline()
        param_grid = self.config["models"][self.model_name]["params"]

        cv = StratifiedKFold(
            n_splits=self.config["cv_folds"],
            shuffle=True,
            random_state=self.config["random_state"],
        )

        logger.info(f"Running GridSearchCV for '{self.model_name}' over {param_grid}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            test_predictions = best_model.predict(X_test)

        logger.info(
            f"Best CV accuracy: {search.best_score_:.4f} (params: {search.best_params_})"
        )
        test_acc = accuracy_score(y_test, test_predictions)
        logger.info(f"Held-out test accuracy: {test_acc:.4f}")
        logger.info("\n" + classification_report(y_test, test_predictions))

        return best_model


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
    model_name: str = "logistic_regression",
    model_path: Path = None,
):

    if model_path == None:
        model_path = MODELS_DIR / f"{model_name}.pkl"

    if model_name not in config["models"]:
        raise typer.BadParameter(
            f"Unknown model '{model_name}', choose from {list(config['models'])}"
        )

    logger.info(f"Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)

    trainer = ModelTrainer(df, config, model_name)
    best_model = trainer.train()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()
