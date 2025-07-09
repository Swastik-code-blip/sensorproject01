import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)

@dataclass
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        try:
            logging.info(f"Connecting to MongoDB: {db_name}.{collection_name}")
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]
            documents = list(collection.find())
            
            if not documents:
                raise CustomException("No documents found in MongoDB collection", sys)

            # Inspect sample document
            logging.info(f"Sample document: {documents[0]}")
            df = pd.DataFrame(documents)
            logging.info(f"Raw DataFrame columns: {df.columns.tolist()}")
            
            if "_id" in df.columns:
                df = df.drop(columns=['_id'], axis=1)
            
            # Define expected columns
            expected_columns = [f"Sensor-{i+1}" for i in range(590)] + ["Good/Bad"]
            if df.columns.tolist() != expected_columns:
                logging.warning(f"Unexpected columns: {df.columns.tolist()}. Attempting to fix.")
                if df.shape[1] != len(expected_columns):
                    raise CustomException(f"Column count mismatch. Expected {len(expected_columns)}, got {df.shape[1]}. Sample document: {documents[0]}", sys)
                df.columns = expected_columns

            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"Processed DataFrame columns: {df.columns.tolist()}")
            logging.info(f"First few rows:\n{df.head().to_string()}")

            return df

        except Exception as e:
            logging.error(f"Error in export_collection_as_dataframe: {str(e)}")
            raise CustomException(e, sys)

    def export_data_into_feature_store_file_path(self) -> str:
        try:
            logging.info("Exporting data from MongoDB")
            raw_file_path = self.data_ingestion_config.artifact_folder
            os.makedirs(raw_file_path, exist_ok=True)
            
            sensor_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME,
                db_name=MONGO_DATABASE_NAME
            )

            feature_store_file_path = os.path.join(raw_file_path, 'wafer_fault.csv')
            logging.info(f"Saving data to {feature_store_file_path}")
            sensor_data.to_csv(feature_store_file_path, index=False)

            logging.info(f"Data saved to {feature_store_file_path}")
            return feature_store_file_path
        except Exception as e:
            logging.error(f"Error in export_data_into_feature_store_file_path: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> str:
        logging.info("Entered initiate_data_ingestion method")
        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()
            logging.info("Got data from MongoDB")
            logging.info("Exiting initiate_data_ingestion method")
            return feature_store_file_path
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise CustomException(e, sys)