from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from src.predict import Predictor
from logger import Logger
from src.database import ClickHouseClient  
from src.vault import get_vault_client
from src.kafka.producer import Producer
import configparser
import os
from dotenv import load_dotenv

SHOW_LOG = True

class Message(BaseModel):
    message: str


class SentimentAPI:
    def __init__(self):
        self.app = FastAPI()
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.predictor = Predictor()
        self.producer = Producer()
        load_dotenv()  # load .env file into environment variables
        
        self.vault_connected = False
        self.vault_client = None
        self._setup_vault() 
        self.kafka_connected = self.producer is not None
        self.db_client = self.vault_client.setup_database("predictions")
        self.db_connected = self.db_client is not None
        self._setup_routes()

    def _setup_vault(self):
        try:
            self.vault_client = get_vault_client()
            if self.vault_client and self.vault_client.is_authenticated():
                self.vault_connected = True
                self.logger.info("Vault client initialized successfully")
            else:
                self.vault_connected = False
                self.logger.error("Failed to initialize Vault client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vault client: {e}")
            self.vault_connected = False

    def _setup_routes(self):
        @self.app.post("/predict/")
        async def predict_sentiment(message: Message):
            try:
                self.logger.info("Received message: %s", message.message)
                result = self.predictor.predict(message.message)
                self.logger.info("Prediction result: %s", result)

                if self.kafka_connected:
                    try:
                        payload = {
                            "message": message.message,
                            "prediction": result,
                            "timestamp": time.time()
                        }
                        self.producer.send(payload)
                        if not payload:
                            self.logger.warning("Failed to send prediction to Kafka")
                    except Exception as e:
                        self.logger.error(f"Exception sending prediction to Kafka: {e}", exc_info=True)
                else:
                    self.logger.warning("Kafka not connected, prediction not sent")
                # try:
                #     self.db_client.insert_data("predictions", message.message, result)
                #     self.logger.info("Saved prediction to DB with message: %s and result: %s", message.message, result)
                # except Exception as e:
                #     self.logger.error(f"Failed to save prediction to DB: {e}")
                self.logger.info(f"Prediction completed: {prediction}")
                if result:
                    return {"sentiment": result}
                else:
                    raise HTTPException(status_code=500, detail="Error during prediction")
            except Exception as e:
                self.logger.error(f"Error in prediction: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")
            
        @self.app.post("/predictions/")
        async def get_predictions(limit: int = 10):
            try:
                if not self.db_connected:
                    raise HTTPException(status_code=503, detail="Database not available")
                self.logger.info(f"Fetching last {limit} predictions from DB")
                rows = self.db_client.get_data("predictions", limit)
                predictions = [
                    {
                        "timestamp": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
                        "message": row[1],
                        "prediction": row[2]
                    }
                    for row in rows
                ]
                return {"predictions": predictions}
            except Exception as e:
                self.logger.error(f"Error fetching predictions: {e}")
                raise HTTPException(status_code=500, detail="Failed to fetch predictions")

        @self.app.get("/vault-status/")
        async def vault_status():
            if not self.vault_connected:
                raise HTTPException(status_code=500, detail="Failed to connect to Vault")
            try:
                vault_status = {
                    "connected": self.vault_connected,
                    "authenticated": self.vault_client.is_authenticated() if self.vault_client else False,
                    "secrets_engine": "Available" if self.vault_client.list_mounted_secrets_engines() else "Not available"
                }
                self.logger.info(f"Vault status: {vault_status}")
                return vault_status
            except Exception as e:
                self.logger.error(f"Error getting Vault status: {e}")
                raise HTTPException(status_code=500, detail="Failed to get Vault status")

        @self.app.get("/kafka-status/")
        async def vault_status():
            """Check Kafka status and connection."""
            if not self.kafka_connected and not self.producer:
                raise HTTPException(status_code=503, detail="Kafka producer not initialized")
            try:
                producer = KafkaProducer(bootstrap_servers="kafka:9092")
                producer.bootstrap_connected()
                producer.close()

                topic_exists = "predictions" in topics
                status = {
                    "connected": True,
                    "topic_exists": topic_exists,
                    "topic": "predictions"
                }
                self.logger.info(f"Kafka status: {status}")
                return status
            except Exception as e:
                logger.error(f"Error checking Kafka status: {e}")
                raise HTTPException(status_code=503, detail=f"Kafka connection error: {str(e)}")
            
        
        @self.app.get("/health/")
        async def health_check():
            status = {
                "status": "healthy",
                "model_loaded": getattr(self, "predictor", None) is not None,
                "database_connected": getattr(self, "db_connected", False),
                "vault_connected": getattr(self, "vault_connected", False),
                "kafka_connected": getattr(self, "kafka_connected", False),
            }
            self.logger.info(f"Health check: {status}")
            return status

        @self.app.get("/ready/")
        async def readiness_check():
            return {"status": "OK"}

    def _on_shutdown(self):
        """Gracefully close producer and other resources"""
        try:
            if getattr(self, "producer", None):
                self.producer.close()
                self.logger.info("Kafka producer closed on shutdown")
        except Exception:
            pass
    
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


def main():
    sentiment_api = SentimentAPI()
    sentiment_api.run()

if __name__ == "__main__":
    main()
