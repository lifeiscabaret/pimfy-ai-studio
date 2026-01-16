from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import io
import base64
from typing import Optional
from PIL import Image
import numpy as np

class BackgroundRequest(BaseModel):
    base64_dog_image: str
    prompt: str
    neg_prompt: Optional[str] = "messy, cluttered, text, letters, blurry, dark, noisy, low quality"
    color_hint: str = "soft pastel"

class BackgroundResponse(BaseModel):
    base64_background_image: str

app = FastAPI(title="SDXL Background Service")
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_models():
    print("SDXL AI 모델 로딩 시작...")
    print("SDXL 로드 완료. (현재는 VRAM 보호를 위해 더미 상태)")

@app.post("/generate/background", response_model=BackgroundResponse)
async def generate_background_api(request: BackgroundRequest):
    try:
        image_data = base64.b64decode(request.base64_dog_image)
        dog_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 image data")

    width, height = dog_image.size
    dummy_background = Image.new('RGB', (width, height), (255, 204, 204)) 
    
    buffered = io.BytesIO()
    dummy_background.save(buffered, format="PNG")
    base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return BackgroundResponse(base64_background_image=base64_img)
