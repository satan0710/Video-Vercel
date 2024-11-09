# style_transfer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
import config
from torchvision.models import vgg19, VGG19_Weights
# FastAPI app initialization
app = FastAPI()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained VGG-19 model
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # Use the updated weights parameter to load the VGG-19 model
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(18)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(27)])
    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3

# Style transfer helper functions
def load_image(image_data: bytes, max_size=512):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(device)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

def style_transfer(content, style, num_steps=300, style_weight=1e6, content_weight=1):
    content_features = vgg(content)
    style_features = vgg(style)

    input_image = content.clone().requires_grad_(True)
    optimizer = optim.LBFGS([input_image])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            input_features = vgg(input_image)

            content_loss = content_weight * torch.mean((input_features[1] - content_features[1]) ** 2)

            style_loss = 0
            for f1, f2 in zip(input_features, style_features):
                G = gram_matrix(f1)
                A = gram_matrix(f2)
                style_loss += torch.mean((G - A) ** 2)

            style_loss *= style_weight
            loss = content_loss + style_loss
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

    return input_image.detach()

# Define VGG model globally for style transfer
vgg = VGGFeatures().to(device)

# FastAPI endpoint for style transfer
@app.post("/style-transfer/")
async def apply_style_transfer(content_file: UploadFile = File(...), style_file: UploadFile = File(...)):
    try:
        # Load content and style images
        content_image = load_image(await content_file.read())
        style_image = load_image(await style_file.read())

        # Perform style transfer
        output = style_transfer(content_image, style_image)
        
        # Post-process and convert output image to PIL format
        output = output.cpu().squeeze(0)
        unloader = transforms.ToPILImage()
        output_image = unloader(output)

        # Save the output image to a buffer
        output_buffer = BytesIO()
        output_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)

        return {"status": "Style transfer applied successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def apply_style_transfer(content_image: Image, style_image: Image) -> Image:
    # Placeholder for GAN-based style transfer
    pass