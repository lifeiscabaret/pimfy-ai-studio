from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import io
import base64
from typing import Optional
from PIL import Image
# from diffusers import StableDiffusionXLImg2ImgPipeline # (1. 현재 SDXL은 사용하지 않는 경우 주석 처리)
import numpy as np

# --- Pydantic 모델 정의 ---
class BackgroundRequest(BaseModel):
    # 메인 서버로부터 전송받을 누끼 딴 강아지 이미지와 배경 프롬프트
    base64_dog_image: str
    prompt: str
    neg_prompt: Optional[str] = "messy, cluttered, text, letters, blurry, dark, noisy, low quality"
    color_hint: str # 예: "pastel pink", "soft blue"

class BackgroundResponse(BaseModel):
    base64_background_image: str

# --- FastAPI 앱 및 AI 모델 변수 선언 ---
app = FastAPI(title="SDXL Background Service")
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_models():
    print("SDXL AI 모델 로딩 시작...")
    print(f"SDXL Using device: {device}")
    
    # models["sdxl_pipe"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     use_safetensors=True
    # ).to(device)
    
    # 🚨 현재 단계에서는 실제 SDXL 로딩 코드 주석 유지 (VRAM 과부하 방지)
    print("SDXL 로드 완료. (현재는 VRAM 보호를 위해 더미 상태)")


@app.post("/generate/background", response_model=BackgroundResponse)
async def generate_background_api(request: BackgroundRequest):
    # 1. Base64 디코딩
    try:
        image_data = base64.b64decode(request.base64_dog_image)
        dog_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 image data")

    # 2. SDXL 프롬프트 구성 (사용자 정의 색상 및 스타일 반영)
    final_prompt = (
        f"A studio portrait background, {request.color_hint} color palette, "
        f"minimalist and clean aesthetic, centered for a dog subject. "
        f"{request.prompt}, professional, 8K."
    )
    
    # 3. ⭐️ SDXL 추론 로직 (더미)
    # 테스트-> 단색의 더미 배경 반환.
    width, height = dog_image.size
    
    # 더미 배경 생성 (파스텔 핑크 예시)
    dummy_background = Image.new('RGB', (width, height), (255, 204, 204)) 
    
    # 4. 합성 및 인코딩
    # (실제 워크플로우에서는 SDXL이 생성한 배경 이미지에 누끼 딴 강아지 이미지를 합성해야 하지만,
    # 이 서버는 배경 이미지만 반환하는 역할이므로, 메인 서버에서 최종 합성을 합니다.)
    
    # 5. Base64 인코딩
    buffered = io.BytesIO()
    dummy_background.save(buffered, format="PNG")
    base64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 6. SDXL 사용 후 VRAM 정리 (선택적)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return BackgroundResponse(base64_background_image=base64_img)
