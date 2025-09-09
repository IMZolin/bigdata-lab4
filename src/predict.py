import argparse
import configparser
from datetime import datetime
import os
import json
from typing import Optional
from fastapi import HTTPException
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import shutil
import sys
import time
import traceback
import yaml

from src.logger import Logger
from src.utils import clean_text, prepare_text
import warnings

warnings.filterwarnings("ignore")

SHOW_LOG = True
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def parse_args():
    parser = argparse.ArgumentParser(description="Predictor")
    parser.add_argument("-t",
        "--tests",
        type=str,
        help="Select tests",
        required=False,
        default="smoke",
        const="smoke",
        nargs="?",
        choices=["smoke", "func"])
    return parser.parse_args()


class Predictor():

    def __init__(self, config_path="config.ini", args=None) -> None:
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.config.read(config_path)
        self.args = args if args is not None else argparse.Namespace(tests="smoke")
        self.logger.info("Predictor is ready")
        try:
            self.X_test = np.load(self.config["SPLIT_DATA"]["X_test"])
            self.y_test = np.load(self.config["SPLIT_DATA"]["y_test"])
        except FileNotFoundError:    
            self.logger.error("File missing.")
            raise HTTPException(status_code=404, detail="File missing")
        except Exception as e:
            self.logger.error(f"Numpy array load failure: {e}")
            raise HTTPException(status_code=500, detail="Numpy array load failure")
        try:
            with open(self.config["NAIVE_BAYES"]["path"], "rb") as model_file:
                self.classifier = pickle.load(model_file)
            with open(self.config["SPLIT_DATA"]["vectorizer"], "rb") as vectorizer_file:
                self.vectorizer = pickle.load(vectorizer_file)
        except FileNotFoundError:
            self.logger.error("Model file not found.")
            raise HTTPException(status_code=404, detail="Model not found")
        except Exception as e:
            self.logger.error(f"Error loading model/vectorizer: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


    def predict(self, message) -> str:
        try:
            if not message:
                self.logger.error("Message is not provided")
                raise HTTPException(
                    status_code=400, 
                    detail="Message is not provided. Please provide a message to analyze."
                )
            cleaned_message = clean_text(message)  
            message_vectorized = self.vectorizer.transform([cleaned_message])  
            sentiment = self.classifier.predict(message_vectorized)
            if sentiment[0] == 1:
                self.logger.info(f"Sentiment for message: '{message}' is Positive")
                return "Positive sentiment"
            elif sentiment[0] == 0:
                self.logger.info(f"Sentiment for message: '{message}' is Negative")
                return "Negative sentiment"
            else:
                self.logger.error(f"Unexpected sentiment value: {sentiment[0]}")
                return "Unknown sentiment"
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction error: {e}"
            )

    def test(self) -> bool:
        if self.args.tests == "smoke":
            self.smoke_test()
        elif self.args.tests == "func":
            self.func_test()
        else:
            self.logger.error("Unknown test type")
            raise HTTPException(status_code=400, detail="Unknown test type")
        return True
        
    def smoke_test(self):
        try:
            print(self.y_test.shape, self.X_test.shape)
            y_pred = self.classifier.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.logger.info(f'Model has {accuracy} Accuracy score')
        except Exception:
            self.logger.error(traceback.format_exc())
            sys.exit(1)
        self.logger.info(f'Model passed smoke tests')

    def func_test(self):
        try:
            tests_path = os.path.join(os.getcwd(), "tests", "test_data")
            exp_path = os.path.join(os.getcwd(), "experiments")
            os.makedirs(exp_path, exist_ok=True)

            for test_file in os.listdir(tests_path):
                try:
                    test_path = os.path.join(tests_path, test_file)
                    with open(test_path) as f:
                        data = json.load(f)
                    
                    X_list = data.get("X")
                    y_list = data.get("y")

                    if not X_list or not y_list:
                        self.log.error(f"Test file {test_file} missing 'X' or 'y'")
                        continue

                    # Extract from the first dictionary
                    X_dict = X_list[0]
                    y_dict = y_list[0]

                    # Sort keys to maintain order
                    X_text = [X_dict[key] for key in sorted(X_dict.keys(), key=int)]
                    y = [y_dict[key] for key in sorted(y_dict.keys(), key=int)]

                    cleaned_X = [clean_text(text) for text in X_text]
                    X_vectorized = self.vectorizer.transform(cleaned_X).toarray()

                    score = self.classifier.score(X_vectorized, y)
                    y_pred = self.classifier.predict(X_vectorized)
                    accuracy = accuracy_score(y, y_pred)
                    report = classification_report(y, y_pred)

                    exp_data = {
                        "score": str(score),
                        "model_path": self.config["NAIVE_BAYES"]["path"],
                        "test_path": test_file,
                        "accuracy": accuracy,
                        "classification_report": report,
                    }

                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test_file[:6]}_{str_date_time}')
                    os.makedirs(exp_dir, exist_ok=True)

                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)

                    # Copy log file if exists
                    log_file = os.path.join(os.getcwd(), "logfile.log")
                    if os.path.exists(log_file):
                        shutil.copy(log_file, os.path.join(exp_dir, "exp_logfile.log"))

                    self.logger.info(f'Model passed func test {test_file}')

                except Exception:
                    self.logger.error(traceback.format_exc())
                    sys.exit(1)

        except Exception as e:
            self.logger.error(f"Error during test: {e}")
            raise HTTPException(status_code=500, detail="Test error")


if __name__ == "__main__":
    args = parse_args()
    predictor = Predictor(args=args, config_path="config.ini")
    predictor.test()