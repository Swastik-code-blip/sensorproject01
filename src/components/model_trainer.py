import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            report = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[model_name] = test_model_score
                logging.info(f"{model_name} - Train score: {train_model_score}, Test score: {test_model_score}")
            return report
        except Exception as e:
            logging.error(f"Error in evaluate_models: {str(e)}")
            raise CustomException(e, sys)

    def get_best_model(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        try:
            model_report = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            logging.info(f"Model report: {model_report}")
            if not model_report:
                raise CustomException("No models evaluated successfully", sys)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model_object = self.models[best_model_name]
            return best_model_name, best_model_object, best_model_score
        except Exception as e:
            logging.error(f"Error in get_best_model: {str(e)}")
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object: object, best_model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> object:
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            logging.info(f"Best params for {best_model_name}: {best_params}")
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
        except Exception as e:
            logging.error(f"Error in finetune_best_model: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Evaluating models")
            model_report = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]
            logging.info(f"Best model: {best_model_name}, Score: {best_model_score}")

            logging.info("Finetuning best model")
            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            logging.info(f"Final best model: {best_model_name}, Score: {best_model_score}")

            if best_model_score < 0.5:
                raise CustomException("No best model found with accuracy >= 0.5", sys)

            logging.info(f"Saving model to {self.model_trainer_config.trained_model_path}")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            return best_model_score
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(e, sys)