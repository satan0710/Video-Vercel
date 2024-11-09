from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse
import traceback

# Import internal utility functions and modules directly
from routers.routes.video_preprocessor import preprocess_video
from routers.routes.scene_detection import detect_scenes
from routers.routes.emotion_detection import predict_emotion
from routers.routes.style_transfer import apply_style_transfer
from routers.routes.branding_overlay import branding_overlay

app = FastAPI()

@app.post("/generate-thumbnail/")
@app.post("/generate-thumbnail/")
async def generate_thumbnail(file: UploadFile = File(...)):
    # Step 1: Receive video and preprocess it to extract frames
    try:
        video_data = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video file")
    
    # Step 2: Send video for preprocessing
    frames = await preprocess_video.process_video(video_data)

    # Step 3: Pass frames through scene detection
    key_frames = await detect_scenes.detect_scenes(frames)

    # Step 4: Analyze each frame's emotion
    emotions = []
    for frame in key_frames:
        emotion = await predict_emotion.predict_emotion(frame)
        emotions.append(emotion)

    # Step 5: Apply style transfer
    styled_frames = []
    for frame in key_frames:
        styled_frame = await apply_style_transfer.apply_style_transfer(frame)
        styled_frames.append(styled_frame)

    # Step 6: Branding overlay on the final styled frame
    final_thumbnail = await branding_overlay.branding_overlay(styled_frames[-1])  # last frame

    # Convert the final result to an image format for return
    buffer = BytesIO()
    final_thumbnail.save(buffer, format="JPEG")
    buffer.seek(0)

    # Return the final thumbnail image
    return StreamingResponse(buffer, media_type="image/jpeg")

# Root endpoint (Optional)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Thumbnail Generator API!"}

# Main entry point to run the app (use this if running directly with `uvicorn`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
