# routers/scene_detection.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from routers.models.scene_detector import SceneDetector,apply_super_resolution
router=APIRouter()
# Initialize the route

# Define input schema for video request
class VideoRequest(BaseModel):
    video_path: str
    detector_type: str = 'content'
    threshold: float = 30.0

@router.post("/detect/")
async def detect_scenes(request: VideoRequest):
    try:
        # Initialize the scene detector with specified parameters
        detector = SceneDetector(detector_type=request.detector_type, threshold=request.threshold)
        
        # Run scene detection and get keyframes
        keyframes = detector.detect_scenes(request.video_path)
        super_res_keyframes = [apply_super_resolution(frame) for frame in keyframes]
        # Return detected keyframes as JSON response
        return {"keyframes": keyframes}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
