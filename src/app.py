from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.predict import Predictor, parse_args
from logger import Logger


class Message(BaseModel):
    message: str

class SentimentAPI:
    def __init__(self, args=None):
        self.app = FastAPI()
        self.logger = Logger(show=True).get_logger(__name__)
        self.predictor = Predictor()
        self.args = args or parse_args()
        self._setup_database()
        self._setup_routes()

    def _setup_database(self):
        """Initialize database connection"""
        pass
    
    def _setup_routes(self):
        @self.app.post("/predict/")
        async def predict_sentiment(message: Message):
            try:
                self.logger.info("Received message: %s", message.message)  
                result = self.predictor.predict(message.message)
                self.logger.info("Prediction result: %s", result)  
                if result:
                    return {"sentiment": result} 
                else:
                    raise HTTPException(status_code=500, detail="Error during prediction")
            except Exception as e:
                self.logger.error(f"Error in prediction: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")

        @self.app.get("/health/")
        async def health_check():
            return {"status": "OK"}
        
    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

def main():
    args = parse_args()
    sentiment_api = SentimentAPI(args)
    sentiment_api.run()

if __name__ == "__main__":
    main()