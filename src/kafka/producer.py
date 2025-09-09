import os
import json
from kafka import KafkaProducer
from src.logger import Logger

SHOW_LOG = True

DEFAULT_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.environ.get("PREDICTIONS_TOPIC", "predictions")

class Producer:
    def __init__(self, bootstrap_servers: str = DEFAULT_BOOTSTRAP, topic: str = TOPIC, retries: int = 5):
        self.logger = Logger(True).get_logger(__name__)
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.retries = retries
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=retries,
            linger_ms=10,
            acks="all"
        )
        
    def send(self, payload: dict, key: str = None, ):
        """Send payload to configured topic. payload must be JSON-serializable."""
        for attempt in range(1, self.retries + 1):
            try:
                future = self.producer.send(self.topic, value=payload, key=(key.encode("utf-8") if key else None))
                result = future.get(timeout=10)
                self.logger.info("Sent message to Kafka topic '%s', partition=%s, offset=%s", self.topic, result.partition, result.offset)
                return True
            except Exception as e:  # pragma: no cover
                self.logger.warning("Kafka send attempt %d/%d failed: %s", attempt, self.retries, e)
                time.sleep(2)
        self.logger.error("Failed to send message to Kafka after %d attempts", self.retries)
        return False

    def close(self):
        try:
            self.producer.flush()
            self.producer.close()
        except Exception:
            pass