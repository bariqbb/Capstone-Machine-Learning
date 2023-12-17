import os
import uvicorn
import traceback
import tensorflow as tf

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array

model = tf.keras.models.load_model('D:/tebak/preprocessingdata1.h5')

app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

class RequestText(BaseModel):
    text:str
    
@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"
        image = load_image_into_numpy_array(uploaded_file.file.read())
        print("Image shape:", image.shape)
        
        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"
    
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)