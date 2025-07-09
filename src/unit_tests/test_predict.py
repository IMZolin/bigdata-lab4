import configparser
import os
import unittest
from unittest.mock import patch, MagicMock
import sys 
import numpy as np
import pickle
from fastapi import HTTPException

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from predict import Predictor

config = configparser.ConfigParser()
config.read("config.ini")

class TestPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = Predictor(args=MagicMock(tests="smoke"))

    def test_init(self):
        assert self.predictor.classifier is not None, (
            "Classifier model was not loaded - check if model file exists and is accessible"
        )
        assert self.predictor.vectorizer is not None, (
            "Text vectorizer was not loaded - check if vectorizer file exists and is accessible"
        )
        assert hasattr(self.predictor, "predict"), (
            "Predictor class is missing required 'predict' method"
        )
        assert hasattr(self.predictor, "test"), (
            "Predictor class is missing required 'test' method"
        )

    # @patch("numpy.load")
    # @patch("pickle.load")
    # def test_predict(self, mock_pickle_load, mock_np_load):
    #     mock_pickle_load.side_effect = [MagicMock(), MagicMock()]
    #     mock_np_load.side_effect = [np.array([1, 2, 3]), np.array([0, 1])]
    #     result = self.predictor.predict("So bad!")
    #     self.assertEqual(result, "Negative sentiment")

    # @patch("numpy.load")
    # @patch("pickle.load")
    # def test_predict_positive(self, mock_pickle_load, mock_np_load):
    #     mock_pickle_load.side_effect = [MagicMock(), MagicMock()]
    #     mock_np_load.side_effect = [np.array([1, 2, 3]), np.array([0, 1])]
    #     result = self.predictor.predict("Great!")
    #     self.assertEqual(result, "Positive sentiment")


    # @patch("numpy.load")
    # @patch("pickle.load")
    # def test_smoke_test(self, mock_pickle_load, mock_np_load):
    #     mock_pickle_load.side_effect = [MagicMock(), MagicMock()]
    #     mock_np_load.side_effect = [np.array([1, 2, 3]), np.array([0, 1])]
    #     result = self.predictor.test()
    #     self.assertTrue(result)

    # @patch("numpy.load")
    # @patch("pickle.load")
    # def test_func_test(self, mock_pickle_load, mock_np_load):
    #     mock_pickle_load.side_effect = [MagicMock(), MagicMock()]
    #     mock_np_load.side_effect = [np.array([1, 2, 3]), np.array([0, 1])]
    #     self.predictor.args.tests = "func"
    #     result = self.predictor.test()
    #     self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()