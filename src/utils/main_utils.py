import sys
from typing import Dict, Tuple
import os
import pandas as pd
import pickle
import yaml
from src.constant import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
        except Exception as e:
            logging.error(f"Error in read_yaml_file: {str(e)}")
            raise CustomException(e, sys)

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))
            return schema_config
        except Exception as e:
            logging.error(f"Error in read_schema_config_file: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info(f"Entered the save_object method of MainUtils class with path: {file_path}")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f"Object saved to {file_path}")
        except Exception as e:
            logging.error(f"Error in save_object: {str(e)}")
            raise CustomException(e, sys)

    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info(f"Entered the load_object method of MainUtils class with path: {file_path}")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No such file or directory: '{file_path}'")
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
            logging.info(f"Object loaded from {file_path}")
            return obj
        except Exception as e:
            logging.error(f"Error in load_object: {str(e)}")
            raise CustomException(e, sys)