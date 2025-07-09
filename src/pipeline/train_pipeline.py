import sys
import os
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging as lg

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            lg.info("Starting data ingestion")
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            lg.info(f"Data ingestion completed. Feature store path: {feature_store_file_path}")
            return feature_store_file_path
        except Exception as e:
            lg.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_store_file_path):
        try:
            lg.info(f"Starting data transformation with file: {feature_store_file_path}")
            datatransformation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr, test_arr, preprocessor_path = datatransformation.initiate_data_transformation()
            lg.info(f"Data transformation completed. Preprocessor path: {preprocessor_path}")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            lg.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

    def start_model_training(self, train_arr, test_arr):
        try:
            lg.info("Starting model training")
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            lg.info(f"Model training completed. Score: {model_score}")
            return model_score
        except Exception as e:
            lg.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            lg.info("Running training pipeline")
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2square = self.start_model_training(train_arr, test_arr)
            lg.info(f"Training completed. Trained model score: {r2square}")
            return r2square
        except Exception as e:
            lg.error(f"Error in pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        lg.error(f"Error in main: {str(e)}")
        raise CustomException(e, sys)