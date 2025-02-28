import configparser
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import sys
import traceback
from src.logger import Logger

SHOW_LOG = True

class Trainer:
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        self.X_train = np.load(self.config["SPLIT_DATA"]["X_train"])
        self.y_train = np.load(self.config["SPLIT_DATA"]["y_train"])
        self.X_test = np.load(self.config["SPLIT_DATA"]["X_test"])
        self.y_test = np.load(self.config["SPLIT_DATA"]["y_test"])

        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.bayes_path = os.path.join(self.project_path, "naive_bayes.pkl")
        self.svm_path = os.path.join(self.project_path, "svm.pkl")
        self.log.info("Trainer is ready")

    def train_naive_bayes(self, predict=False, alpha=1.0, fit_prior=True):
        classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)    
        if predict:
            y_pred = classifier.predict(self.X_test)
            self.log.info(f"Naive Bayes Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            self.log.info("Classification Report (Naive Bayes):\n" + classification_report(self.y_test, y_pred))
        params = {'path': self.bayes_path, 'alpha': alpha, 'fit_prior': fit_prior}
        return self.save_model(classifier, self.bayes_path, "NAIVE_BAYES", params)


    def save_model(self, classifier, path:str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_naive_bayes(predict=True, alpha=1.0, fit_prior=True)
