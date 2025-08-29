import configparser
import glob
import os
import shutil
import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys 
import numpy as np
import pickle
from fastapi import HTTPException

from src.logger import Logger

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from predict import Predictor

SHOW_LOG = True

class TestPredictor(unittest.TestCase):
    def setUp(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.args = MagicMock(tests="smoke")
        self.predictor = Predictor(args=self.args, config_path="test_config.ini")
        try:
            self.classifier = self.predictor.classifier
            self.vectorizer = self.predictor.vectorizer
        except Exception:
            raise HTTPException(status_code=500)
        self.args = MagicMock(tests="smoke")
        self.mock_classifier = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.predictor.classifier = self.mock_classifier
        self.predictor.vectorizer = self.mock_vectorizer
        self.predictor.X_test = np.array([[1, 2]])
        self.predictor.y_test = np.array([1])

    def tearDown(self):
        exp_dir = os.path.join(os.getcwd(), "experiments")
        for folder in glob.glob(os.path.join(exp_dir, "exp_test*")):
            if os.path.isdir(folder):
                shutil.rmtree(folder)


    def test_parse_args_defaults(self):
        with patch("sys.argv", ["predict.py"]):
            from predict import parse_args
            args = parse_args()
            self.assertEqual(args.tests, "smoke")

    def test_parse_args_func(self):
        with patch("sys.argv", ["predict.py", "--tests", "func"]):
            from predict import parse_args
            args = parse_args()
            self.assertEqual(args.tests, "func")

    def test_init(self):
        assert self.predictor.classifier is not None
        assert self.predictor.vectorizer is not None
        assert hasattr(self.predictor, "predict")
        assert hasattr(self.predictor, "test")

    @patch("pickle.load", side_effect=Exception("Unexpected error"))
    def test_init_unexpected_exception(self, mock_pickle):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.args)
        self.assertEqual(context.exception.status_code, 500)

    def test_no_message(self):
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict(message=None)
        exception = context.exception
        self.assertEqual(exception.status_code, 400)
        self.assertIn("Message is not provided", str(exception.detail))

    def test_empty_string(self):
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict(message="")
        exception = context.exception
        self.assertEqual(exception.status_code, 400)
        self.assertIn("Message is not provided", str(exception.detail))

    def test_unexpected_sentiment_value(self):
        self.mock_vectorizer.transform.return_value.toarray.return_value = np.array([[1, 2]])
        self.mock_classifier.predict.return_value = [42]  
        result = self.predictor.predict("Some neutral message")
        self.assertEqual(result, "Unknown sentiment")

    def test_predict_vectorizer_error(self):
        self.mock_vectorizer.transform.side_effect = Exception("Vectorizer failed")
        with self.assertRaises(HTTPException) as context:
            self.predictor.predict("I like it")
        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("Prediction error", str(context.exception.detail))

    def test_positive_message(self):
        dummy_input = "I love this product!"
        self.mock_classifier.predict.return_value = [1]
        self.mock_vectorizer.transform.return_value.toarray.return_value = np.array([[1, 2]])
        prediction = self.predictor.predict(dummy_input)
        self.assertEqual(prediction, "Positive sentiment")

    def test_negative_message(self):
        dummy_input = "I hate this product!"
        self.mock_classifier.predict.return_value = [0]
        self.mock_vectorizer.transform.return_value.toarray.return_value = np.array([[1, 2]])
        prediction = self.predictor.predict(dummy_input)
        self.assertEqual(prediction, "Negative sentiment")

    def test_smoke_test_success(self):
        self.mock_classifier.predict.return_value = np.array([0])
        with patch("src.predict.accuracy_score", return_value=0.85):
            self.predictor.smoke_test()
            self.mock_classifier.predict.assert_called_once()

    def test_smoke_test(self):
        with patch("sys.argv", ["predict.py", "--test", "smoke"]):
            self.predictor.smoke_test = MagicMock()
            self.predictor.test()
            self.predictor.smoke_test.assert_called_once()

    def test_func_test(self):
        self.predictor.args.tests = "func"
        self.predictor.func_test = MagicMock()
        self.predictor.test()
        self.predictor.func_test.assert_called_once()

    @patch("shutil.copy")  
    @patch("os.listdir", return_value=["test.json"])
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    @patch("builtins.open", new_callable=mock_open, read_data='{"X": [{"0": "I love it"}], "y": [{"0": 1}]}')
    def test_func_test_success(self, mock_file, mock_join, mock_listdir, mock_copy):
        self.mock_vectorizer.transform.return_value.toarray.return_value = np.array([[1, 2]])
        self.mock_classifier.score.return_value = 0.9
        self.mock_classifier.predict.return_value = [1]
        self.predictor.func_test()
        self.mock_classifier.score.assert_called_once()
        mock_copy.assert_called_once()  

    def test_unknown_test_type(self):
        self.predictor.args.tests = "invalid"
        with self.assertRaises(HTTPException) as context:
            self.predictor.test()
        self.assertEqual(context.exception.status_code, 400)

    def test_smoke_test_failure(self):
        self.mock_classifier.score.side_effect = Exception("Scoring failed")
        with patch("sys.exit") as mock_exit:
            self.predictor.smoke_test()
            mock_exit.assert_called_once_with(1)

    @patch("builtins.open", new_callable=mock_open, read_data='{"bad_json": ')
    def test_func_test_json_error(self, mock_file):
        with patch("os.listdir", return_value=["dummy.json"]), \
            patch("os.path.join", side_effect=lambda *args: "/".join(args)), \
            patch("sys.exit") as mock_exit:
            self.predictor.func_test()
            mock_exit.assert_called_once()

    def test_predict_all_sentiments(self):
        self.mock_vectorizer.transform.return_value.toarray.return_value = np.array([[1, 2]])
        for label, expected in zip([0, 1], ["Negative sentiment", "Positive sentiment"]):
            self.mock_classifier.predict.return_value = [label]
            result = self.predictor.predict("sample")
            self.assertEqual(result, expected)

    def test_func_test_outer_exception(self):
        with patch("os.path.join", side_effect=Exception("Outer error")):
            with self.assertRaises(HTTPException) as context:
                self.predictor.func_test()
            self.assertEqual(context.exception.status_code, 500)
            self.assertIn("Test error", str(context.exception.detail))

    @patch("os.listdir", return_value=["test.json"])
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_func_test_inner_exception(self, mock_join, mock_listdir):
        with patch("builtins.open", side_effect=Exception("File read error")), patch("sys.exit") as mock_exit:
            self.predictor.func_test()
            mock_exit.assert_called_once()


    @patch("numpy.load", side_effect=Exception("Numpy array load failure"))
    def test_init_np_load_unexpected_exception(self, mock_np_load):
        with self.assertRaises(HTTPException) as context:
            Predictor(self.args)
        self.assertEqual(context.exception.status_code, 500)

if __name__ == "__main__":
    Logger(SHOW_LOG).get_logger(__name__).info("TEST PREDICT IS READY")
    unittest.main()