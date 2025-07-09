import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging as lg
import pickle

class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.artifact_folder = "artifacts"
        self.target_column = "Good/Bad"

    def initiate_data_transformation(self):
        try:
            lg.info(f"Loading data from {self.feature_store_file_path}")
            if not os.path.exists(self.feature_store_file_path):
                raise FileNotFoundError(f"CSV file not found at {self.feature_store_file_path}")
            
            with open(self.feature_store_file_path, 'r') as file:
                lg.info("First 5 lines of CSV:")
                for i, line in enumerate(file):
                    if i < 5:
                        lg.info(line.strip())
                    if i >= 5:
                        break

            df = pd.read_csv(self.feature_store_file_path, header=0)
            lg.info("DataFrame columns: %s", df.columns.tolist())
            
            if df.columns[0] == '0' or isinstance(df.columns[0], int):
                lg.warning("Numeric columns detected. Assigning expected columns.")
                expected_columns = [f"Sensor-{i+1}" for i in range(df.shape[1]-1)] + [self.target_column]
                if df.shape[1] != len(expected_columns):
                    raise CustomException(f"Column count mismatch. Expected {len(expected_columns)}, got {df.shape[1]}", sys)
                df.columns = expected_columns
                lg.info("Assigned columns: %s", df.columns.tolist())

            if self.target_column not in df.columns:
                raise CustomException(f"Target column '{self.target_column}' not found in DataFrame. Available columns: {list(df.columns)}", sys)

            lg.info("First few rows:\n%s", df.head().to_string())
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column].map({1: 1, -1: 0})

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            train_path = os.path.join(self.artifact_folder, "train.csv")
            test_path = os.path.join(self.artifact_folder, "test.csv")
            pd.DataFrame(train_arr, columns=X.columns.tolist() + [self.target_column]).to_csv(train_path, index=False)
            pd.DataFrame(test_arr, columns=X.columns.tolist() + [self.target_column]).to_csv(test_path, index=False)

            scaler_path = os.path.join(self.artifact_folder, "scaler.pkl")
            with open(scaler_path, "wb") as file:
                pickle.dump(scaler, file)

            lg.info(f"Data transformation completed. Train path: {train_path}, Test path: {test_path}, Scaler path: {scaler_path}")
            return train_arr, test_arr, scaler_path

        except Exception as e:
            lg.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        data_path = os.path.join("artifacts", "wafer_fault.csv")
        transformer = DataTransformation(feature_store_file_path=data_path)
        train_arr, test_arr, scaler_path = transformer.initiate_data_transformation()
        lg.info(f"Transformation complete. Train shape: {train_arr.shape}, Test shape: {test_arr.shape}, Scaler: {scaler_path}")
    except Exception as e:
        lg.error(f"Error in main: {str(e)}")
        raise CustomException(e, sys)