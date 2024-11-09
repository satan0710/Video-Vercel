# branding_overlay.py

import torch
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO
model = YOLO('yolov5s.pt')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model with the local path to YOLOv5


 # Adjust for your YOLOv5 model and device

# Insert the project root directory into sys.path
PROJECT_DIR = Path("C:/thumbnail_generator")
sys.path.insert(0, str(PROJECT_DIR))

import config

FONT_PATH = config.FONT if hasattr(config, 'FONT') else "arial.ttf"


# FastAPI app initialization
app = FastAPI()

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Overlay configuration
FONT_SIZE = 20
BRANDING_TEXT = "Your Brand Here"

def detect_and_overlay(image: Image.Image, branding_text: str = BRANDING_TEXT) -> Image.Image:
    # Convert image to format suitable for YOLO model (RGB)
    img = image.convert("RGB")
    
    # Perform inference with YOLOv5
    results = model([img])
    detections = results.pandas().xyxy[0]  # Get bounding boxes from YOLO
    
    # Draw bounding boxes and branding overlay
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()
    
    for _, row in detections.iterrows():
        # Get bounding box coordinates and confidence score
        x1, y1, x2, y2, conf, class_id = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])
        
        # Draw the bounding box
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
        
        # Place branding text just above the bounding box
        text_position = (x1, y1 - FONT_SIZE)
        draw.text(text_position, branding_text, fill="red", font=font)

    return img

@app.post("/branding-overlay/")
async def branding_overlay(file: UploadFile = File(...)):
    # Load the uploaded image
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Apply branding overlay
    processed_image = detect_and_overlay(image)

    # Save the result in a BytesIO buffer for response
    buffer = BytesIO()
    processed_image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Return the image with overlay as a streaming response
    return StreamingResponse(buffer, media_type="image/jpeg")
def apply_background_inpainting(image: Image, mask: np.array) -> Image:
    # Placeholder for background inpainting GAN
    pass