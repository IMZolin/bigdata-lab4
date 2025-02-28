from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.preprocess import DataMaker
from src.train import Trainer
from src.predict import Predictor 
import logging

app = FastAPI()

class Message(BaseModel):
    message: str

data_maker = DataMaker()
data_maker.split_data()
trainer = Trainer()
trainer.train_naive_bayes(predict=True, alpha=1.0, fit_prior=True)
predictor = Predictor()

@app.post("/predict/")
async def predict_sentiment(message: Message):
    try:
        result = predictor.predict_sentiment(message.message)
        if result:
            return {"sentiment": result} 
        else:
            raise HTTPException(status_code=500, detail="Error during prediction")
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health/")
async def health_check():
    return {"status": "OK"}
