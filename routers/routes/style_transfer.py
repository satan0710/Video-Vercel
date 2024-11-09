# routers/style_transfer.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from routers.models.style_transfer import load_image, style_transfer, apply_style_transfer
import torch
import torchvision.transforms as transforms
router=APIRouter()
# Initialize the router

@router.post("/transfer/")
async def apply_style_transfer(content_file: UploadFile = File(...), style_file: UploadFile = File(...)):
    try:
        # Load content and style images
        content_image = load_image(await content_file.read())
        style_image = load_image(await style_file.read())

        # Perform style transfer
        output = style_transfer(content_image, style_image)
        
        # Convert output to a PIL image for response
        output = output.cpu().squeeze(0)
        unloader = transforms.ToPILImage()
        output_image = unloader(output)
        #GAN Placeholder
        styled_image = apply_style_transfer(content_image, style_image)
        # Save the result in a buffer for return
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        buffer.seek(0)

        return {"status": "Style transfer applied successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
