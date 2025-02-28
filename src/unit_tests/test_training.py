import configparser
import os
import unittest
import numpy as np
from unittest import mock
import sys
from src.train import Trainer

config = configparser.ConfigParser()
config.read("config.ini")

class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.trainer = Trainer()

    def test_train_naive_bayes(self):
        """Test the Naive Bayes model training and prediction"""
        with mock.patch("sklearn.naive_bayes.MultinomialNB.fit") as mock_fit, \
             mock.patch("sklearn.naive_bayes.MultinomialNB.predict") as mock_predict:
            mock_fit.return_value = None
            mock_predict.return_value = self.mock_y_test
            self.trainer.train_naive_bayes(predict=True, alpha=1.0, fit_prior=True)
            mock_fit.assert_called_once()
            mock_predict.assert_called_once()
            self.assertEqual(mock_predict.return_value.tolist(), self.mock_y_test.tolist())

if __name__ == "__main__":
    unittest.main()
