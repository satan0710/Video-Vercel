# routers/emotion_detection.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from routers.models.emotion_detector import EmotionNet,apply_expression_manipulation
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms

# Initialize router
router = APIRouter()

# Load and configure the model
model = EmotionNet(num_classes=7)  # Adjust for your number of emotion classes
model.load_state_dict(torch.load(r"C:\thumbnail_generator\emotion_detector_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define transformation for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@router.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()

        # Convert image data to a PIL image
        image = Image.open(BytesIO(image_data))

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        manipulated_image = apply_expression_manipulation(image, "joy")
        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)

        # Map the prediction to an emotion label
        predicted_emotion = emotion_labels[predicted_class.item()]
        return {"predicted_class": predicted_emotion}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
