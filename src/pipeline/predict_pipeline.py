import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predictions_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')  # Fixed typo
    preprocessor_path: str = os.path.join(artifact_folder, 'scaler.pkl')  # Fixed typo
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

class PredictionPipeline:
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.predictions_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = 'prediction_artifacts'
            os.makedirs(pred_file_input_dir, exist_ok=True)
            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)
            logging.info(f"Input file saved to {pred_file_path}")
            return pred_file_path
        except Exception as e:
            logging.error(f"Error in save_input_files: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            logging.info(f"Loading model from {self.predictions_pipeline_config.model_file_path}")
            model = self.utils.load_object(self.predictions_pipeline_config.model_file_path)
            logging.info(f"Loading preprocessor from {self.predictions_pipeline_config.preprocessor_path}")
            preprocessor = self.utils.load_object(self.predictions_pipeline_config.preprocessor_path)
            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)  # Fixed: Added transformed_x
            return preds
        except Exception as e:
            logging.error(f"Error in predict: {str(e)}")
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            logging.info(f"Input DataFrame columns: {input_dataframe.columns.tolist()}")

            # Handle unexpected columns
            expected_columns = [f"Sensor-{i+1}" for i in range(590)]
            if input_dataframe.columns.tolist()[:590] != expected_columns:
                logging.warning(f"Unexpected columns: {input_dataframe.columns.tolist()}. Attempting to fix.")
                if len(input_dataframe.columns) < 590:
                    raise CustomException(f"Expected at least {len(expected_columns)} columns, got {len(input_dataframe.columns)}", sys)
                input_dataframe = input_dataframe.iloc[:, :590]
                input_dataframe.columns = expected_columns

            # Drop unnamed columns if present
            input_dataframe = input_dataframe.drop(columns=[col for col in input_dataframe.columns if col.startswith("Unnamed")], errors='ignore')

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = predictions
            target_column_mapping = {0: 'Bad', 1: 'Good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.predictions_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.predictions_pipeline_config.prediction_file_path, index=False)
            logging.info(f"Predictions saved to {self.predictions_pipeline_config.prediction_file_path}")

            return self.predictions_pipeline_config
        except Exception as e:
            logging.error(f"Error in get_predicted_dataframe: {str(e)}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting prediction pipeline")
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            return self.predictions_pipeline_config
        except Exception as e:
            logging.error(f"Error in run_pipeline: {str(e)}")
            raise CustomException(e, sys)