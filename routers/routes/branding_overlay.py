# router.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import ImageDraw,ImageFont,Image
from io import BytesIO
import torch
from ultralytics import YOLO
import config
from pathlib import Path
import sys
from routers.models.branding_overlay import apply_background_inpainting
branding_router=APIRouter()
# Create a router instance

# Load the YOLO model
model = YOLO('yolov5s.pt')

# Set device for YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure font path and branding text
FONT_PATH = config.FONT if hasattr(config, 'FONT') else "arial.ttf"
FONT_SIZE = 20
BRANDING_TEXT = "Your Brand Here"

# Define the detection and overlay function
def detect_and_overlay(image: Image.Image, branding_text: str = BRANDING_TEXT) -> Image.Image:
    img = image.convert("RGB")
    results = model([img])
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes and branding overlay
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()
    
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, class_id = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
        text_position = (x1, y1 - FONT_SIZE)
        draw.text(text_position, branding_text, fill="red", font=font)

    return img

# Define the branding overlay route

async def branding_overlay(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    processed_image = detect_and_overlay(image)
    buffer = BytesIO()
    inpainted_image = apply_background_inpainting(image, None)
    processed_image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/jpeg")
