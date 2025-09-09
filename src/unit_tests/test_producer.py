
import unittest
from unittest.mock import patch, MagicMock
import json
from src.kafka.producer import Producer

class TestProducer(unittest.TestCase):
    @patch("src.kafka.producer.Producer")
    def test_producer_send_success(self, mock_kafka_producer):
        mock_future = MagicMock()
        mock_future.get.return_value.partition = 0
        mock_future.get.return_value.offset = 1
        mock_producer_instance = MagicMock()
        mock_producer_instance.send.return_value = mock_future
        mock_kafka_producer.return_value = mock_producer_instance

        producer = Producer()
        payload = {"message": "test"}
        result = producer.send(payload)
        self.assertTrue(result)
        mock_producer_instance.send.assert_called_once()
        mock_future.get.assert_called_once_with(timeout=10)

    @patch("src.kafka.producer.Producer")
    def test_producer_send_failure(self, mock_kafka_producer):
        mock_producer_instance = MagicMock()
        mock_producer_instance.send.side_effect = Exception("Kafka down")
        mock_kafka_producer.return_value = mock_producer_instance

        producer = Producer(retries=2)
        payload = {"message": "fail_test"}
        result = producer.send(payload)
        self.assertFalse(result)
        self.assertEqual(mock_producer_instance.send.call_count, 2)

    @patch("src.kafka_module.KafkaProducer")
    def test_producer_close(self, mock_kafka_producer):
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance
        producer = Producer()
        producer.close()
        mock_producer_instance.flush.assert_called_once()
        mock_producer_instance.close.assert_called_once()



if __name__ == "__main__":
    unittest.main()