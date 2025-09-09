import unittest
from unittest.mock import patch, MagicMock
import json
from src.kafka.consumer import Consumer


class TestConsumer(unittest.TestCase):
    @patch("src.kafka.consumer.Consumer")
    @patch("src.kafka.consumer.get_vault_client")
    def test_consumer_initialization(self, mock_get_vault, mock_kafka_consumer):
        # Mock Vault client and DB
        mock_db = MagicMock()
        mock_vault = MagicMock()
        mock_vault.setup_database.return_value = mock_db
        mock_get_vault.return_value = mock_vault

        consumer_instance = Consumer()
        self.assertEqual(consumer_instance.topic, "predictions")
        self.assertEqual(consumer_instance.group_id, "prediction-group")
        self.assertTrue(hasattr(consumer_instance.consumer, "__iter__"))

    def test_handle_message_inserts_to_db(self):
        mock_db = MagicMock()
        consumer = Consumer.__new__(Consumer)
        consumer.db_client = mock_db
        consumer.logger = MagicMock()

        # Test with standard message
        msg = {"message": "Hello", "prediction": {"sentiment": "positive"}}
        consumer._handle_message(msg)
        mock_db.insert_data.assert_called_once_with(
            "predictions", "Hello", {"sentiment": "positive"}
        )

        # Test with only sentiment key
        mock_db.reset_mock()
        msg = {"sentiment": "neutral"}
        consumer._handle_message(msg)
        mock_db.insert_data.assert_called_once_with(
            "predictions", str(msg), "neutral"
        )


if __name__ == "__main__":
    unittest.main()