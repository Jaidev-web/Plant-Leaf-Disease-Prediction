from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import asyncio
import os
import io
import json
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_IMAGE_SIZE = (240, 240)

# 1. Load the dynamic class names from the JSON file
with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

# 2. Build the EfficientNetB1 Architecture
def build_model():
    base_model = EfficientNetB1(
        weights=None, # We are loading our own weights
        include_top=False, 
        input_shape=(*TARGET_IMAGE_SIZE, 3) # EfficientNetB1 specific input size
    )
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(len(class_names), activation='softmax')
    ])
    return model

# 3. Load the weights safely
try:
    model = build_model()
    weights_path = os.path.join(BASE_DIR, 'efficientnetb1_plant_final.weights.h5')
    model.load_weights(weights_path)
    print("EfficientNetB1 Weights Loaded Successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    model = None

# 4. Process image directly from memory
def pred_plant_disease(image_bytes):
    # Read the bytes into a PIL Image and resize
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise ValueError("Invalid or corrupted image file.")
        
    img = img.resize(TARGET_IMAGE_SIZE) 
    
    print("@@ Got Image for prediction")
    
    test_image = img_to_array(img) / 255.0 # Normalize 0-1
    test_image = np.expand_dims(test_image, axis=0)
    
    result = model.predict(test_image)
    
    # Get the index of the highest probability
    pred_index = np.argmax(result, axis=1)[0]
    
    # Match the index to the class_names.json list
    predicted_disease = class_names[str(pred_index)] 
    
    return predicted_disease

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"status": "AgroVision EfficientNet API is running successfully!"}
    
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model failed to load on server."})

    print("@@ Input posted = ", image.filename)
    
    # Read the file data into memory
    contents = await image.read()

    print("@@ Predicting class......")
    try:
        # Offload the synchronous CPU-bound prediction to a separate thread
        prediction_result = await asyncio.to_thread(pred_plant_disease, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
          
    return JSONResponse(content={"prediction": prediction_result})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)