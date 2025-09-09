import json
import os
import time
import signal
import threading
from kafka import KafkaConsumer
from src.logger import Logger
from src.vault import get_vault_client
from src.database import ClickHouseClient


SHOW_LOG = True

DEFAULT_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.environ.get("PREDICTIONS_TOPIC", "predictions")
GROUP_ID = os.environ.get("KAFKA_CONSUMER_GROUP", "prediction-group")

from kafka.errors import NoBrokersAvailable

class Consumer:
    def __init__(self, bootstrap_servers=DEFAULT_BOOTSTRAP, topic=TOPIC, group_id=GROUP_ID):
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._stopped = threading.Event()

        self.vault_client = get_vault_client()
        self.db_client = self.vault_client.setup_database("predictions")

        self.logger.info(f"Initializing Kafka consumer: {self.bootstrap_servers} topic={self.topic} group={self.group_id}")

        retries = 5
        for attempt in range(1, retries + 1):
            try:
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset="earliest",
                    enable_auto_commit=True,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
                )
                self.logger.info("Kafka consumer connected successfully")
                break
            except NoBrokersAvailable as e:
                self.logger.warning(f"Kafka not available (attempt {attempt}/{retries}): {e}")
                time.sleep(5)
        else:
            raise RuntimeError("Failed to connect to Kafka after retries")

    def _handle_message(self, msg_value):
        """
        Expected message format: {'message': 'text', 'prediction': {...}} or similar.
        We attempt to extract fields message/prediction; if not present we store raw payload.
        """
        try:
            message = msg_value.get("message") if isinstance(msg_value, dict) else None
            prediction = msg_value.get("prediction") if isinstance(msg_value, dict) else None

            # fallback: if top-level 'sentiment' exists, map it
            if prediction is None and "sentiment" in msg_value:
                prediction = msg_value.get("sentiment")
            if hasattr(self.db_client, "insert_data"):
                self.db_client.insert_data("predictions", message or str(msg_value), prediction or str(msg_value))
            else:
                self.logger.warning("ClickHouse client has no insert_data method; message skipped: %s", msg_value)
            self.logger.info("Saved message to ClickHouse: %s", msg_value)
        except Exception as e:  # pragma: no cover
            self.logger.error("Error handling incoming message: %s", e, exc_info=True)    

    def run(self):
        self.logger.info("Starting Kafka consumer loop...")
        for msg in self.consumer:
            if self._stopped.is_set():
                break
            try:
                payload = msg.value
                self.logger.info(f"Received message from Kafka: {payload}")
                if self.db_client:
                    try:
                        self.db_client.insert_data(
                            "predictions",
                            payload.get("message"),
                            payload.get("prediction")
                        )
                        self.logger.info(f"Saved to DB: {payload}")
                    except Exception as e:
                        self.logger.error(f"Failed to save to DB: {e}", exc_info=True)
                self.logger.info(f"Saved to ClickHouse: {msg.value}")
            except Exception: # pragma: no cover
                self.logger.error(f"Error processing message: {e}", exc_info=True)

    def stop(self):
        self._stopped.set()
        if hasattr(self, "consumer"):
            self.consumer.close()
        self.logger.info("Kafka consumer stopped (topic=%s)", self.topic)



if __name__ == "__main__": # pragma: no cover
    consumer = Consumer()
    consumer.run()