import warnings
from pathlib import Path
import mlflow
from mlflow import MlflowClient, MlflowException
import os
import dagshub
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score,ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import LogTransformer, save_file, save_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, param_config, REPORTS_DIR,PROJ_ROOT,FIGURES_DIR




app = typer.Typer()

dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    mlflow=True,
)

mlflow.set_experiment(param_config["mlflow_experiment"])

class ModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split_data(self):
        X = self.df.drop(columns=[param_config["target_col"]])
        y = self.df[param_config["target_col"]]
        return train_test_split(
            X,
            y,
            test_size=param_config["test_size"],
            random_state=param_config["random_state"],
            stratify=y,
        )

    def build_pipeline(self, clf) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("log", LogTransformer()),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, param_config["numeric_cols"]),
                ("binary", "passthrough", param_config["binary_cols"]),
            ]
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    def build_classifier(self, classifier_name, trial):

        params = param_config["models"][classifier_name]["params"]

        if classifier_name == "logistic_regression":
            C = trial.suggest_float(
                "lr_c", low=params["C"]["low"], high=params["C"]["high"], log= params["C"]["log"]
            )
            penalty = trial.suggest_categorical(
                "lr_penalty", list(params["penalty"]["choices"])
            )
            return LogisticRegression(C=C, solver="liblinear", penalty=penalty)

        elif classifier_name == "random_forest":
            n_estimators = trial.suggest_int(
                "rf_n_estimators",
                low=params["n_estimators"]["low"],
                high=params["n_estimators"]["high"],
            )
            max_depth = trial.suggest_int(
                "rf_max_depth",
                low=params["max_depth"]["low"],
                high=params["max_depth"]["high"],
            )
            min_samples_split= trial.suggest_int(
                "rf_min_samples_split",
                low= params["min_samples_split"]["low"],
                high= params["min_samples_split"]["low"]
            )
            
            min_samples_leaf= trial.suggest_int(
                "rf_min_samples_leaf",
                low= params["min_samples_leaf"]["low"],
                high= params["min_samples_leaf"]["low"]
            )
            
            max_features= trial.suggest_float(
                "rf_max_features",
                low= params["max_features"]["low"],
                high= params["max_features"]["low"]
            )                                    
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,max_features=max_features
            )
        
        elif classifier_name == "xgboost":
            n_estimators = trial.suggest_int(
                "xgb_n_estimators",
                low=params["n_estimators"]["low"],
                high=params["n_estimators"]["high"],
            )

            max_depth = trial.suggest_int(
            "xgb_max_depth",
                low=params["max_depth"]["low"],
                high=params["max_depth"]["high"],
            )

            learning_rate = trial.suggest_float(
                "xgb_learning_rate",
                low=params["learning_rate"]["low"],
                high=params["learning_rate"]["high"],
                log=True,
            )

            min_child_weight = trial.suggest_int(
                "xgb_min_child_weight",
                low=params["min_child_weight"]["low"],
                high=params["min_child_weight"]["high"],
            )

            subsample = trial.suggest_float(
                "xgb_subsample",
                low=params["subsample"]["low"],
                high=params["subsample"]["high"],
            )

            colsample_bytree = trial.suggest_float(
                "xgb_colsample_bytree",
                low=params["colsample_bytree"]["low"],
                high=params["colsample_bytree"]["high"],
            )

            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=param_config["random_state"],
                eval_metric="logloss",
            )
            
            

        raise ValueError(f"Unknown classifer ;{classifier_name}")

    def objective(self, trial):
        classifier_name = trial.suggest_categorical(
            "classifier", list(param_config["models"].keys())
        )
        clf = self.build_classifier(classifier_name, trial)
        pipeline = self.build_pipeline(clf)

        cv = StratifiedKFold(
            n_splits=param_config["cv_folds"],
            shuffle=True,
            random_state=param_config["random_state"],
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train, cv=cv, scoring="f1"
            )
        mean_score = scores.mean()
        std= scores.std()

        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_f1", mean_score)
            mlflow.log_metric("cv_f1_std", std)
            

        return mean_score

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

        data_dir = PROCESSED_DATA_DIR
        datas = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }
        for key, value in datas.items():
            save_csv(data_dir, value, key)
            
        with mlflow.start_run(run_name="optuna_search"):
            study= optuna.create_study(direction="maximize")
            study.optimize(self.objective,param_config["n_trials"])
            
            best_params = study.best_params
            classifier_name = best_params["classifier"]
            logger.info(f"Best trial: {best_params} (cv_accuracy={study.best_value:.4f})")
            best_trial = optuna.trial.FixedTrial(best_params)
            clf = self.build_classifier(classifier_name, best_trial)
            best_pipeline = self.build_pipeline(clf)
            
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=RuntimeWarning)
                best_pipeline.fit(self.X_train, self.y_train)
            

            test_predictions = best_pipeline.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, test_predictions)
            test_f1 = f1_score(self.y_test, test_predictions)
            test_recall = recall_score(self.y_test, test_predictions)
            test_precision = precision_score(self.y_test, test_predictions)
            
            #classification report
            clf_report= classification_report(self.y_test,test_predictions)
            clf_reports_save_path= REPORTS_DIR/"classification.txt"
            clf_reports_save_path.parent.mkdir(exist_ok=True)
            save_file(clf_reports_save_path,clf_report)
            
            #confussion matrix
            cm_display= ConfusionMatrixDisplay.from_predictions(self.y_test, test_predictions)
            cm_display.ax_.set_title(f"Confusion matris {classifier_name}")
            cm_display.figure_.tight_layout()
            cm_display_file_path=FIGURES_DIR/"confusion_matrix.png"
            cm_display_file_path.parent.mkdir(exist_ok=True)
            cm_display.figure_.savefig(cm_display_file_path)
            
            #optimization graph
            optimize_graph= plot_optimization_history(study=study)
            optimize_graph_save_path= FIGURES_DIR/"optimize_graph.png"
            optimize_graph_save_path.parent.mkdir(exist_ok=True)
            optimize_graph.figure.savefig(optimize_graph_save_path)
            
            

            mlflow.log_param("best_classifier", classifier_name)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_precision", test_precision)
            
            
            model_info= mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                name= f'{classifier_name}',
                serialization_format="pickle",
                registered_model_name= param_config["mlflow_model_name"], 
                input_example=self.X_train.head()
                )
            client= MlflowClient()
            try:
                champion= client.get_model_version_by_alias(param_config["mlflow_model_name"],"champion")
                champion_f1= client.get_run(champion.run_id).data.metrics["test_f1"]
            except MlflowException:
                champion_f1 = None
            
            if champion_f1 is None or test_f1> champion_f1:
                client.set_registered_model_alias(param_config["mlflow_model_name"], "champion", model_info.registered_model_version)
                logger.success(f"Promoted version {model_info.registered_model_version} to champion (f1={test_f1:.4f})")
            else:
                logger.info(f"New model (f1={test_f1:.4f}) did not beat champion (f1={champion_f1:.4f}) â not promoted")
                
                
            
            mlflow.log_artifact(clf_reports_save_path, artifact_path="reports")
            mlflow.log_artifact(cm_display_file_path, artifact_path="figures")
            mlflow.log_artifact(optimize_graph_save_path, artifact_path="figures")
            mlflow.log_artifact(str(PROJ_ROOT/"param_config.yaml"), artifact_path="configuration")
    

            logger.info(f"Held-out test accuracy: {test_acc:.4f}")
            logger.info("\n" + clf_report)

        return best_pipeline, classifier_name
            
            
        
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
    model_path: Path = None
):
    logger.info(f"Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)

    trainer = ModelTrainer(df)
    best_model, classifier_name = trainer.train()

    if model_path is None:
        model_path = MODELS_DIR /"model.pkl"
        
    

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.success(f"Model saved to {model_path}")


if __name__ == "__main__":
    app()