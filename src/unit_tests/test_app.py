import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from app import SentimentAPI

SHOW_LOG = False


class TestSentimentAPI(unittest.TestCase):
    @patch("src.predict.Predictor")  
    @patch("src.kafka.producer.Producer")  
    @patch("app.get_vault_client")
    def setUp(self, mock_vault_client, mock_producer_class, mock_predictor_class):
        self.mock_predictor = mock_predictor_class.return_value
        self.mock_predictor.predict.return_value = "Positive sentiment"

        self.mock_producer = mock_producer_class.return_value
        self.mock_producer.send.return_value = True

        self.mock_vault = mock_vault_client.return_value
        self.mock_vault.is_authenticated.return_value = True
        self.mock_vault.list_mounted_secrets_engines.return_value = {"secret/": {}}
        self.mock_vault.setup_database.return_value = MagicMock(
            get_data=MagicMock(
                return_value=[("2025-09-05T12:00:00", "Hello world", "Positive sentiment")]
            ),
            insert_data=MagicMock(return_value=True)
        )

        # Initialize API
        self.api_instance = SentimentAPI()
        self.client = TestClient(self.api_instance.app)

    def test_readiness_check(self):
        response = self.client.get("/ready/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})

    def test_health_check(self):
        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["model_loaded"])
        self.assertTrue(data["database_connected"])
        self.assertTrue(data["vault_connected"])
        self.assertTrue(data["kafka_connected"])

    def test_predict_success(self):
        payload = {"message": "I love this!"}
        response = self.client.post("/predict/", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"sentiment": "Positive sentiment"})
        self.api_instance.predictor.predict.assert_called_once_with("I love this!")
        self.api_instance.producer.send.assert_called_once()

    def test_predict_none_result(self):
        self.api_instance.predictor.predict.return_value = None
        payload = {"message": "Hello"}
        response = self.client.post("/predict/", json=payload)
        self.assertEqual(response.status_code, 500)
        self.assertIn("Prediction failed", response.text)

    def test_get_predictions_success(self):
        response = self.client.post("/predictions/", json={})
        self.assertEqual(response.status_code, 200)
        preds = response.json()["predictions"]
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["message"], "Hello world")

    def test_get_predictions_db_error(self):
        self.api_instance.db_client.get_data.side_effect = Exception("DB fail")
        response = self.client.post("/predictions/", json={})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to fetch predictions", response.text)

    def test_vault_status_success(self):
        response = self.client.get("/vault-status/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["connected"])
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["secrets_engine"], "Available")

    def test_vault_status_not_connected(self):
        self.api_instance.vault_connected = False
        response = self.client.get("/vault-status/")
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to connect to Vault", response.text)

    def test_vault_status_exception(self):
        self.api_instance.vault_client.is_authenticated.side_effect = Exception("Vault fail")
        response = self.client.get("/vault-status/")
        self.assertEqual(response.status_code, 500)
        self.assertIn("Failed to get Vault status", response.text)

    def test_on_shutdown_closes_producer(self):
        mock_close = MagicMock()
        self.api_instance.producer.close = mock_close
        self.api_instance._on_shutdown()
        mock_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
