import cv2
import scenedetect
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
# FastAPI app initialization
app = FastAPI()

class SceneDetector:
    def __init__(self, detector_type='content', threshold=30.0):
        self.scene_manager = scenedetect.SceneManager()

        if detector_type == 'content':
            self.scene_manager.add_detector(scenedetect.detectors.ContentDetector(threshold=threshold))
        elif detector_type == 'threshold':
            self.scene_manager.add_detector(scenedetect.detectors.ThresholdDetector(threshold=threshold))
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Use 'content' or 'threshold'.")

    def detect_scenes(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found.")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file '{video_path}'.")

        self.scene_manager.detect_scenes(frame_source=cap)
        scenes = self.scene_manager.get_scene_list()
        
        keyframes = [scene[0].get_start_time().get_seconds() for scene in scenes]
        cap.release()
        return keyframes

# Input schema using Pydantic for video path
class VideoRequest(BaseModel):
    video_path: str
    detector_type: str = 'content'
    threshold: float = 30.0

@app.post("/detect-scenes/")
async def detect_scenes(request: VideoRequest):
    try:
        detector = SceneDetector(detector_type=request.detector_type, threshold=request.threshold)
        keyframes = detector.detect_scenes(request.video_path)
        return {"keyframes": keyframes}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
def apply_super_resolution(image: Image) -> Image:
    # Placeholder for super-resolution GAN
    pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#curl -X 'POST' \'http://127.0.0.1:8000/detect-scenes/' \-H 'Content-Type: application/json' \-d '{video_path": "C:/videos/sample.mp4","detector_type": "content","threshold": 30.0}'