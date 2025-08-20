import configparser
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys 
import numpy as np
import pickle
from fastapi import HTTPException

from src.logger import Logger

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from predict import Predictor

config = configparser.ConfigParser()
config.read("config.ini")
SHOW_LOG = True

class TestPredictor(unittest.TestCase):
    def setUp(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = os.path.join(self.temp_dir.name, "config.ini")
        with open(self.temp_config_path, 'w') as f:
            config.write(f)
        self.predictor = Predictor(config_path=self.temp_config_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        self.assertIsNotNone(self.predictor.classifier)
        self.assertIsNotNone(self.predictor.vectorizer)
        self.assertIsNotNone(self.predictor.X_test)
        self.assertIsNotNone(self.predictor.y_test)

    @patch("numpy.load", side_effect=FileNotFoundError("File missing"))
    def test_init_np_file_missing(self, mock_np_load):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.temp_config_path)
        self.assertEqual(context.exception.status_code, 404)

    @patch("numpy.load", side_effect=Exception("Array load failed"))
    def test_init_np_unexpected_error(self, mock_np_load):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.temp_config_path)
        self.assertEqual(context.exception.status_code, 500)

    @patch("pickle.load", side_effect=FileNotFoundError("Model not found"))
    def test_init_pickle_file_missing(self, mock_pickle):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.temp_config_path)
        self.assertEqual(context.exception.status_code, 404)

    @patch("pickle.load", side_effect=Exception("Internal error"))
    def test_init_pickle_unexpected_error(self, mock_pickle):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.temp_config_path)
        self.assertEqual(context.exception.status_code, 500)
 
    def test_predict_positive_negative(self):
        messages = ["I love this product!", "I hate this product!"]
        for msg in messages:
            result = self.predictor.predict(msg)
            self.assertIn(result, ["Positive sentiment", "Negative sentiment"])

    def test_no_message(self):
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict(None)
        self.assertEqual(context.exception.status_code, 400)
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict("")
        self.assertEqual(context.exception.status_code, 400)

    def test_predict_unexpected_sentiment_value(self):
        self.predictor.vectorizer.transform = MagicMock(return_value=np.array([[1, 2]]))
        self.predictor.classifier.predict = MagicMock(return_value=[42])  
        result = self.predictor.predict("neutral text")
        self.assertEqual(result, "Unknown sentiment")

    def test_predict_vectorizer_error(self):
        self.predictor.vectorizer.transform = MagicMock(side_effect=Exception("Vectorizer failed"))
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict("some text")
        self.assertEqual(context.exception.status_code, 500)

    def test_test_smoke(self):
        self.predictor.smoke_test = MagicMock()
        self.predictor.args.tests = "smoke"
        self.assertTrue(self.predictor.test())
        self.predictor.smoke_test.assert_called_once()

    def test_test_func(self):
        self.predictor.func_test = MagicMock()
        self.predictor.args.tests = "func"
        self.assertTrue(self.predictor.test())
        self.predictor.func_test.assert_called_once()

    def test_test_invalid_type(self):
        self.predictor.args.tests = "invalid"
        with self.assertRaises(HTTPException) as context:
            self.predictor.test()
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    Logger(SHOW_LOG).get_logger(__name__).info("TEST PREDICT IS READY")
    unittest.main()