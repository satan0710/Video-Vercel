import cv2
import numpy as np
from typing import Optional, List
from PIL import Image
from io import BytesIO
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
import torch
from pathlib import Path
from tempfile import NamedTemporaryFile


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load face detection classifier once (for optional face cropping)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create FastAPI router
router = APIRouter()

def preprocess_video(video_path, interval=1, target_size=(256, 256), grayscale=False, face_detection=False, 
                     blur: Optional[str] = None, color_space: Optional[str] = None, 
                     dynamic_frame_selection=False, sharpen=False) -> List[Image.Image]:
    """
    Preprocess video by extracting and processing frames with configurable options.
    """
    with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_path)
        tmp_file_path = tmp_file.name  # Get the file path of the temporary file

    frames = []
    cap = cv2.VideoCapture(tmp_file_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {tmp_file_path}")
        return frames
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        logging.debug(f"Processing frame {frame_count}")
        
        # Dynamic frame selection based on scene change if enabled
        if dynamic_frame_selection and prev_frame is not None:
            if not detect_scene_change(prev_frame, frame):
                prev_frame = frame
                frame_count += 1
                continue  # Skip this frame if no significant change detected

        # Process every nth frame based on interval
        if frame_count % (interval * int(fps)) == 0:
            try:
                # Preprocess single frame
                processed_frame = preprocess_single_frame(
                    frame, target_size=target_size, grayscale=grayscale, 
                    face_detection=face_detection, blur=blur, color_space=color_space, sharpen=sharpen
                )
                frames.append(processed_frame)
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")

        prev_frame = frame
        frame_count += 1

    cap.release()
    logging.info(f"Total processed frames: {len(frames)}")
    return frames


def preprocess_single_frame(frame, target_size=(256, 256), grayscale=False, face_detection=False, 
                            blur=None, color_space: Optional[str] = None, sharpen=False):
    """
    Apply individual preprocessing steps to a single video frame.
    """
    # Resize the frame to the target size
    frame = cv2.resize(frame, target_size)

    # Convert to grayscale if specified
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection and cropping (optional)
    if face_detection:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            frame = frame[y:y+h, x:x+w]  # Crop to the first detected face
            frame = cv2.resize(frame, target_size)  # Resize again to target size

    # Convert to specified color space
    if color_space == 'HSV':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Apply optional blurring
    if blur == 'gaussian':
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    elif blur == 'median':
        frame = cv2.medianBlur(frame, 5)

    # Apply sharpening if specified
    if sharpen:
        frame = apply_sharpening(frame)

    # Normalize pixel values to [0, 1]
    frame = frame / 255.0

    # Convert to RGB format for compatibility
    frame_pil = Image.fromarray((frame * 255).astype(np.uint8)).convert("RGB")
    return frame_pil


def detect_scene_change(prev_frame, current_frame, threshold=0.6):
    """
    Detects significant scene changes between two frames using histogram comparison.
    """
    prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    current_hist = cv2.calcHist([cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    hist_diff = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
    return hist_diff < threshold  # Returns True if scene change is detected


def apply_sharpening(frame):
    """
    Applies sharpening to enhance details in the frame.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Laplacian sharpening kernel
    return cv2.filter2D(frame, -1, kernel)


@router.post("/process_video/")
async def process_video(file: UploadFile = File(...), interval: int = 1, target_size: Optional[str] = "256,256",
                        grayscale: bool = False, face_detection: bool = False, blur: Optional[str] = None, 
                        color_space: Optional[str] = None, dynamic_frame_selection: bool = False, sharpen: bool = False):
    """
    API endpoint for video preprocessing.
    """
    # Parse target size
    width, height = map(int, target_size.split(","))
    
    try:
        # Save uploaded video temporarily
        video_path = f"/tmp/{file.filename}"
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # Process video frames
        frames = preprocess_video(
            video_path, interval=interval, target_size=(width, height), grayscale=grayscale, 
            face_detection=face_detection, blur=blur, color_space=color_space, 
            dynamic_frame_selection=dynamic_frame_selection, sharpen=sharpen
        )

        # Create a BytesIO object for each frame for StreamingResponse
        def image_stream():
            for frame in frames:
                buffer = BytesIO()
                frame.save(buffer, format="JPEG")
                buffer.seek(0)
                yield buffer.getvalue()

        # Return frames as a streaming response
        return StreamingResponse(image_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    except Exception as e:
        logging.error(f"Error in video processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during video processing.")
