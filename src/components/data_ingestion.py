# Importing the required libraries:

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


# Defining class variables without the init method using dataclass:
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")


# creating class for data ingestion:
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the Data Ingestion Component.")
            
            logging.info("Reading gemstone.csv file.")
            df = pd.read_csv("notebook/data/gemstone.csv")
            logging.info("Completed reading of gemstone.csv as pandas dataframe.")

            logging.info("Creating directory to store raw data.")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info("Created raw data directory.")

            logging.info("Saving dataframe to the desired data path.")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data to the desired path.")

            logging.info("Splitting the raw data into train and test files.")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train Test split completed.")

            logging.info("Saving train data to train data path.")
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            logging.info("Saved train data to train data path.")

            logging.info("Saving test data to test data path.")
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Saved test data to test data path.")

            logging.info("Data Ingestion Completed.")

            return(self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e,sys)


# run data ingestion:

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initate_model_training(train_arr, test_arr))







