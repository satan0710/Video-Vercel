from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
app = FastAPI()
# Assuming EmotionNet is your model class
class EmotionNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 6 * 6)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load your trained model
model = EmotionNet(num_classes=7)  # Adjust based on the number of emotion classes
model.load_state_dict(torch.load(r"C:\thumbnail_generator\emotion_detector_model.pth",weights_only=True))
model.eval()

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

@app.post("/predict_emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_data = await file.read()

        # Convert image data to a PIL image
        image = Image.open(BytesIO(image_data))

        # Preprocess the image (resize, convert to grayscale, and tensor)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)

        # Return the predicted class (you can map this to emotion labels if needed)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        return {"predicted_class": emotion_labels[predicted_class.item()]}

    except Exception as e:
        return {"error": str(e)}
def apply_expression_manipulation(image: Image, target_expression: str) -> Image:
    # Placeholder for GAN-based facial expression manipulation
    pass
