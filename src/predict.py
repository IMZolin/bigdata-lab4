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

    def __init__(self, args) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.args = args
        self.log.info("Predictor is ready")
        self.X_test = np.load(self.config["SPLIT_DATA"]["x_test"])
        self.y_test = np.load(self.config["SPLIT_DATA"]["y_test"])
        try:
            self.classifier = pickle.load(open(self.config["NAIVE_BAYES"]["path"], "rb"))
            self.vectorizer = pickle.load(open(self.config["SPLIT_DATA"]["vectorizer"], "rb"))
        except FileNotFoundError:
            self.log.error("Model file not found.")
            raise HTTPException(status_code=404, detail="Model not found")
        except Exception as e:
            self.log.error(f"Error loading model/vectorizer: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


    def predict(self, message) -> str:
        try:
            if not message:
                self.log.error("Message is not provided")
                raise HTTPException(
                    status_code=400, 
                    detail="Message is not provided. Please provide a message to analyze."
                )
            cleaned_message = clean_text(message)  
            message_vectorized = self.vectorizer.transform([cleaned_message]).toarray()  
            sentiment = self.classifier.predict(message_vectorized)
            if sentiment[0] == 1:
                self.log.info(f"Sentiment for message: '{message}' is Positive")
                return "Positive sentiment"
            elif sentiment[0] == 0:
                self.log.info(f"Sentiment for message: '{message}' is Negative")
                return "Negative sentiment"
            else:
                self.log.error(f"Unexpected sentiment value: {sentiment[0]}")
                return "Unknown sentiment"
        except HTTPException:
            raise
        except Exception as e:
            self.log.error(f"Error during prediction: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction error: {e}"
            )

    def test(self) -> bool:
        if self.args.tests == "smoke":
            self.smoke_test()
        elif self.args.tests == "func":
            self.func_test()
        return True
        
    def smoke_test(self):
        try:
            score = self.classifier.score(self.X_test, self.y_test)
            self.log.info(f'Model has {score} score')
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        self.log.info(f'Model passed smoke tests')

    def func_test(self):
        try:
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")

            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        X_dict = data["X"][0]
                        y_dict = data["y"][0]

                        X_text = [X_dict[key] for key in sorted(X_dict.keys(), key=int)]
                        y = [y_dict[key] for key in sorted(y_dict.keys(), key=int)]

                        cleaned_X = [clean_text(text) for text in X_text]
                        X_vectorized = self.vectorizer.transform(cleaned_X).toarray()
                        score = self.classifier.score(X_vectorized, y)
                        self.log.info(f'Test has {score:.3f} score')

                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)

                    self.log.info(f'Model passed func test {f.name}')
                    exp_data = {
                        "score": str(score),
                        "model_path": self.config["NAIVE_BAYES"]["path"],
                        "test_path": test,
                    }

                    y_pred = self.classifier.predict(X_vectorized)
                    accuracy = accuracy_score(y, y_pred)
                    report = classification_report(y, y_pred)
                    exp_data['accuracy'] = accuracy
                    exp_data['classification_report'] = report

                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    self.log.info('TESTS PASSED')
                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))

        except Exception as e:
            self.log.error(f"Error during test: {e}")
            raise HTTPException(status_code=500, detail="Test error")


if __name__ == "__main__":
    args = parse_args()
    predictor = Predictor(args)
    predictor.test()
