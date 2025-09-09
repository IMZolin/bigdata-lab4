import configparser
import os
import numpy as np
import pandas as pd
import pickle
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import Logger 
from src.utils import clean_text, prepare_text

TEST_SIZE = 0.2
SHOW_LOG = True

class DataMaker:
    def __init__(self, config_path="config.ini", project_path=None) -> None:
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        if project_path:
            self.project_path = project_path
        else:
            self.project_path = os.path.join(os.getcwd(), "data")
        os.makedirs(self.project_path, exist_ok=True)

        self.data_path = os.path.join(self.project_path, "data.csv")
        self.train_path = [
            os.path.join(self.project_path, "Train_X.npy"),
            os.path.join(self.project_path, "Train_y.npy"),
        ]
        self.test_path = [
            os.path.join(self.project_path, "Test_X.npy"),
            os.path.join(self.project_path, "Test_y.npy"),
        ]
        self.vectorizer_path = os.path.join(self.project_path, "vectorizer.pkl")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.logger.info("DataMaker is ready")

    def split_data(self, test_size=TEST_SIZE) -> bool:
        if not os.path.isfile(self.data_path):
            self.logger.error(f"Training data file {self.data_path} not found!")
            return False
        dataset = pd.read_csv(self.data_path, encoding="ISO-8859-1")
        dataset["SentimentText"] = dataset["SentimentText"].astype(str).apply(clean_text)
        
        X_tfidf = prepare_text(dataset["SentimentText"], self.vectorizer)
        y = dataset[["Sentiment"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=test_size, random_state=42
        )

        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])

        rel_train_x = os.path.relpath(self.train_path[0], start=os.getcwd())
        rel_train_y = os.path.relpath(self.train_path[1], start=os.getcwd())
        rel_test_x = os.path.relpath(self.test_path[0], start=os.getcwd())
        rel_test_y = os.path.relpath(self.test_path[1], start=os.getcwd())
        rel_vectorizer = os.path.relpath(self.vectorizer_path, start=os.getcwd())

        with open(self.vectorizer_path, "wb") as vec_file:
            pickle.dump(self.vectorizer, vec_file)
        
        self.config["SPLIT_DATA"] = {
            "X_train": rel_train_x,
            "y_train": rel_train_y,
            "X_test": rel_test_x,
            "y_test": rel_test_y,
            "vectorizer": rel_vectorizer,
        }
        self.logger.info("Train and test data is ready")
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)
        return all(os.path.isfile(path) for path in self.train_path + self.test_path)

    def save_splitted_data(self, df: np.ndarray, path: str) -> bool:
        np.save(path, df)  
        self.logger.info(f"{path} is saved")
        return os.path.isfile(path)
    

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()