from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import tensorflow as tf
from prediction import predict

load_dotenv()

model = tf.keras.models.load_model('my-model.keras')
app = FastAPI()

origins = os.getenv("ALLOWED_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/{n_days}")
async def Predict(n_days: int):
    predict_value = predict(model, n_days).tolist()
    return {"predict_value": predict_value}
