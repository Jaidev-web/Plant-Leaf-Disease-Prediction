#Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import os
import io
from PIL import Image

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(BASE_DIR, 'model.h5')

try:
    model = load_model(filepath)
    print("Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def pred_tomato_dieas(image_bytes):
  img = Image.open(io.BytesIO(image_bytes))
  if img.mode != 'RGB':
      img = img.convert('RGB')
  test_image = img.resize((128, 128)) # load image in memory
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "Tomato - Bacteria Spot Disease"
       
  elif pred==1:
      return "Tomato - Early Blight Disease"
        
  elif pred==2:
      return "Tomato - Healthy and Fresh"
        
  elif pred==3:
      return "Tomato - Late Blight Disease"
       
  elif pred==4:
      return "Tomato - Leaf Mold Disease"
        
  elif pred==5:
      return "Tomato - Septoria Leaf Spot Disease"
        
  elif pred==6:
      return "Tomato - Target Spot Disease"
        
  elif pred==7:
      return "Tomato - Tomoato Yellow Leaf Curl Virus Disease"
  elif pred==8:
      return "Tomato - Tomato Mosaic Virus Disease"
        
  elif pred==9:
      return "Tomato - Two Spotted Spider Mite Disease"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def home():
    return {"message": "Welcome to the Plant Leaf Disease Prediction API!"}
    
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    filename = image.filename        
    print("@@ Input posted = ", filename)
    
    image_bytes = await image.read()

    print("@@ Predicting class......")
    prediction_result = pred_tomato_dieas(image_bytes)
          
    return JSONResponse(content={"prediction": prediction_result})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
    
