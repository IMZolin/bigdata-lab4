import configparser
import os
import pandas as pd
import re
import string
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger import Logger 

TEST_SIZE = 0.2
SHOW_LOG = True

class DataMaker:
    def __init__(self) -> None:
        self.logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = self.logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        os.makedirs(self.project_path, exist_ok=True)
        self.data_path = os.path.join(self.project_path, "data.csv")
        self.train_path = [
            os.path.join(self.project_path, "Train_X.csv"),
            os.path.join(self.project_path, "Train_y.csv"),
        ]
        self.test_path = [
            os.path.join(self.project_path, "Test_X.csv"),
            os.path.join(self.project_path, "Test_y.csv"),
        ]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.log.info("DataMaker is ready")

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", "", text)  
        text = re.sub(r"@\w+", "", text)  
        text = re.sub(r"#\w+", "", text)  
        text = text.translate(str.maketrans("", "", string.punctuation))  
        return text.lower().strip()

    def split_data(self, test_size=TEST_SIZE) -> bool:
        if not os.path.isfile(self.data_path):
            self.log.error(f"Training data file {self.data_path} not found!")
            return False
        dataset = pd.read_csv(self.data_path, encoding="ISO-8859-1")
        dataset["SentimentText"] = dataset["SentimentText"].astype(str).apply(self.clean_text)
        
        X_tfidf = self.vectorizer.fit_transform(dataset["SentimentText"])
        X = pd.DataFrame(X_tfidf.toarray())
        y = dataset[["Sentiment"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config["SPLIT_DATA"] = {
            "X_train": self.train_path[0],
            "y_train": self.train_path[1],
            "X_test": self.test_path[0],
            "y_test": self.test_path[1],
        }
        self.log.info("Train and test data is ready")
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)
        return all(os.path.isfile(path) for path in self.train_path + self.test_path)

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df.to_csv(path, index=False)
        self.log.info(f"{path} is saved")
        return os.path.isfile(path)

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
