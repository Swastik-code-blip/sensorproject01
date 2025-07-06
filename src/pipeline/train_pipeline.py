import sys
import os

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            feature_store_file_path=data_ingestion.initiate_data_ingestion
            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,feature_store_file_path):
        try:
            datatransformation=DataTransformation(feature_store_file_path=feature_store_file_path )
            train_arr,test_arr,preprocessor_path=datatransformation.initiate_data_transformation()
            return train_arr, test_arr,preprocessor_path
        except Exception as e:
            raise CustomException(e,sys)
    def start_model_traning(self,train_arr,test_arr):
        try:
            model_trainer= ModelTrainer()

            model_score=model_trainer.initiate_model_trainer(
                train_arr,test_arr            )
            
            return model_score
        except Exception as e:
            raise CustomException(e,sys)
    def run_pipeline(self):
        try:
            feature_store_file_path=self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.datatransformation.initiate_data_transformation()
            r2square=self.start_model_traning(train_arr,test_arr)

            print("traning compleates . Trained model score: ",r2square )
        
        except Exception as e:
            raise CustomException(e,sys)

