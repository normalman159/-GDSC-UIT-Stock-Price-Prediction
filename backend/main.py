from fastapi import FastAPI
import tensorflow as tf
from prediction import predict

model = tf.keras.models.load_model('my-model.keras')
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/{n_days}")
async def Predict(n_days: int):
    predict_value = predict(model, n_days).tolist()
    return {"predict_value": predict_value}
