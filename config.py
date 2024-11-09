# config.py

from pathlib import Path

# Define the base project directory
PROJECT_DIR = Path("C:/thumbnail_generator")

# Paths for storing temporary files and processed thumbnails
TEMP_DIR = PROJECT_DIR / "temp"
THUMBNAILS_DIR = PROJECT_DIR / "thumbnails"

# Model file paths (adjust these paths once models are available or downloaded)
EMOTION_MODEL_PATH = r'C:\thumbnail_generator\emotion_detector_model.pth'
STYLE_TRANSFER_MODEL_PATH = r"C:\thumbnail_generator\vgg19-dcbb9e9d.pth"
YOLO_MODEL_PATH = PROJECT_DIR / "models/yolov5_model.pth"
# Additional configurations
FRAME_WIDTH = 256    # Width to resize frames if needed
FRAME_HEIGHT = 256   # Height to resize frames if needed
FONT = "arial.ttf"   # Font file for text overlays on thumbnails (if used)

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# Model parameters and settings can be added here as needed
EMOTION_MODEL_THRESHOLD = 0.8    # Confidence threshold for emotion detection
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for branding/logo detection

#thumbnail_generator/
#├── main.py                # FastAPI entry point
#├── config.py              # Configuration settings for models and paths
#├── models/
#│   ├── emotion_detector.py    # EmotionNet-based emotion detection
#│   ├── scene_detector.py      # Keyframe extraction using PySceneDetect
#│   ├── style_transfer.py      # VGG-19 style transfer
#│   └── branding_overlay.py    # YOLOv5 logo/branding overlay
#└── utils/
#    └── video_processing.py    # Utility functions for video processing
