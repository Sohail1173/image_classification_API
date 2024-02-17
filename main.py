from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI,UploadFile


def load_model():
    modelpath=r"C:\Users\91808\Downloads\image_classification_API\best.pt"
    model=YOLO(modelpath)
    return model

model=load_model()

app=FastAPI()

@app.post('/get_predictions')
async def get_prediction(file:UploadFile):
    image=await file.read()
    image=Image.open(io.BytesIO(image))
    result=model(image)
    names=result[0].names
    probability=result[0].probs.data.numpy()
    Prediction=np.argmax(probability)
    response={
        "Prediction":names[Prediction],
        "Confidence":float(probability[Prediction])
    }
    return response
