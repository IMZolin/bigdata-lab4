import configparser
import os
import unittest
from unittest.mock import patch, MagicMock
import sys 
import numpy as np
import pickle
import pytest
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
        self.predictor = Predictor(args=MagicMock(tests="smoke"))
        self.mock_classifier = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.predictor.classifier = self.mock_classifier
        self.predictor.vectorizer = self.mock_vectorizer

    def test_init(self):
        assert self.predictor.classifier is not None
        assert self.predictor.vectorizer is not None
        assert hasattr(self.predictor, "predict")
        assert hasattr(self.predictor, "test")

    def test_no_message(self):
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict(message=None)
        exception = context.exception
        self.assertEqual(exception.status_code, 400)
        self.assertIn("Message is not provided", str(exception.detail))

    def test_empty_string(self):
        """Test edge case: empty string message"""
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict(message="")
        exception = context.exception
        self.assertEqual(exception.status_code, 400)
        self.assertIn("Message is not provided", str(exception.detail))

    def test_positive_message(self):
        """Test prediction with valid message"""
        dummy_input = "I love this product!"
        prediction = self.predictor.predict(dummy_input)
        assert isinstance(prediction[0], str), "The prediction output should be string"

    def test_negative_message(self):
        """Test prediction with valid message"""
        dummy_input = "I hate this product!"
        prediction = self.predictor.predict(dummy_input)
        assert isinstance(prediction[0], str), "The prediction output should be string"

    def test_smoke_test(self):
        """Ensure smoke test method returns expected True"""
        with patch("sys.argv", ["predict.py", "--test", "smoke"]):
            assert self.predictor.test() == True, "Smoke test failed"

    def test_func_test(self):
        """Ensure func tests method returns expected True"""
        with patch("sys.argv", ["predict.py", "--test", "func"]):
            assert self.predictor.test() == True, "Func tests failed"


if __name__ == "__main__":
    Logger(SHOW_LOG).get_logger(__name__).info("TEST PREDICT IS READY")
    unittest.main()