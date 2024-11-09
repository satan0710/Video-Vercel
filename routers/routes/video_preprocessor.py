from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO
from routers.models.video_preprocessing import preprocess_video  # Assuming you have this in your utils
from typing import List
from PIL import Image
from fastapi.responses import JSONResponse
import logging

# Initialize the router
router = APIRouter()

# Endpoint to upload a video, preprocess and extract key frames
@router.post("/process-video/")
async def process_video(file: UploadFile = File(...), interval: int = 1, target_size: tuple = (256, 256)):
    try:
        # Read the video file content into memory
        video_data = await file.read()
        video_bytes = BytesIO(video_data)

        # Preprocess the video and extract frames (you can customize this function based on your logic)
        frames = preprocess_video(video_bytes, interval=interval, target_size=target_size)

        # Convert frames to a list of image objects or metadata
        frame_images = []
        for frame in frames:
            # Convert each frame (PIL Image) to a base64 string or other format if needed
            frame_images.append(frame)

        return JSONResponse(content={"message": "Video processed successfully", "frame_count": len(frame_images)})

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
