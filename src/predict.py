import argparse
import configparser
from datetime import datetime
import os
import json
from fastapi import HTTPException
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shutil
import sys
import time
import traceback
import yaml

from src.logger import Logger
from src.utils import clean_text, prepare_text

SHOW_LOG = True


class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m",
                                 "--mode",
                                 type=str,
                                 help="Select mode",
                                 required=True,
                                 default="predict",
                                 const="predict",
                                 nargs="?",
                                 choices=["predict", "test"])
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=False,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])
        self.parser.add_argument("-msg", 
                                 "--message", 
                                 type=str, 
                                 help="Input Twitter message for sentiment prediction", 
                                 required=False)
        # self.vectorizer = pickle.load(open(self.config["SPLIT_DATA"]["vectorizer"], "rb"))
        self.X_train = np.load(self.config["SPLIT_DATA"]["X_train"])
        self.y_train = np.load(self.config["SPLIT_DATA"]["y_train"])
        self.X_test = np.load(self.config["SPLIT_DATA"]["X_test"])
        self.y_test = np.load(self.config["SPLIT_DATA"]["y_test"])
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()
        if args.mode == "predict":
            if args.message:
                self.predict_sentiment(args.message)
            else:
                self.log.error("Please provide a Twitter message for prediction using -msg or --message argument.")
                sys.exit(1)
        else:
            if args.tests == "smoke":
                self.smoke_test(args)
            elif args.tests == "func":
                self.func_test(args)
        return True
    
    def predict_sentiment(self, message: str) -> str:
        try:
            classifier = pickle.load(open(self.config["NAIVE_BAYES"]["path"], "rb"))
            vectorizer = pickle.load(open(self.config["SPLIT_DATA"]["vectorizer"], "rb"))
        except FileNotFoundError:
            self.log.error("Model file not found.")
            raise HTTPException(status_code=404, detail="Model not found")
        except Exception as e:
            self.log.error(f"Error loading model/vectorizer: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        try:
            cleaned_message = clean_text(message)  
            message_vectorized = vectorizer.transform([cleaned_message]).toarray()  
            sentiment = classifier.predict(message_vectorized)
            if sentiment[0] == 1:
                self.log.info(f"Sentiment for message: '{message}' is Positive")
                return "Positive sentiment"
            elif sentiment[0] == 0:
                self.log.info(f"Sentiment for message: '{message}' is Negative")
                return "Negative sentiment"
            else:
                self.log.error(f"Unexpected sentiment value: {sentiment[0]}")
                return "Unknown sentiment"
        except Exception as e:
            self.log.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail="Prediction error")
        
    def smoke_test(self, args):
        try:
            classifier = pickle.load(open(self.config["NAIVE_BAYES"]["path"], "rb"))
            score = classifier.score(self.X_test, self.y_test)
            print(f'{args.model} has {score} score')
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        self.log.info(f'{self.config[args.model]["path"]} passed smoke tests')

    def func_test(self, args):
        tests_path = os.path.join(os.getcwd(), "tests")
        exp_path = os.path.join(os.getcwd(), "experiments")
        classifier = pickle.load(open(self.config[args.model]["path"], "rb"))
        for test in os.listdir(tests_path):
            with open(os.path.join(tests_path, test)) as f:
                try:
                    data = json.load(f)
                    X = self.sc.transform(pd.json_normalize(data, record_path=['X']))
                    y = pd.json_normalize(data, record_path=['y'])
                    score = classifier.score(X, y)
                    print(f'{args.model} has {score} score')
                except Exception:
                    self.log.error(traceback.format_exc())
                    sys.exit(1)
                self.log.info(f'{self.config[args.model]["path"]} passed func test {f.name}')
                exp_data = {
                    "model": args.model,
                    "model params": dict(self.config.items(args.model)),
                    "tests": args.tests,
                    "score": str(score),
                    "X_test path": self.config["SPLIT_DATA"]["x_test"],
                    "y_test path": self.config["SPLIT_DATA"]["y_test"],
                }
                date_time = datetime.fromtimestamp(time.time())
                str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                os.mkdir(exp_dir)
                with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                    yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))
                shutil.copy(self.config[args.model]["path"], os.path.join(exp_dir, f'exp_{args.model}.pkl'))


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
