# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from io import BytesIO

# --- 1. 기본 설정 및 AI 모델 로딩 준비 ---
app = FastAPI()

# Pydantic 모델: 프론트엔드에서 어떤 정보를 받을지 정의합니다. (API의 메뉴판)
class ProfileRequest(BaseModel):
    name: str
    breed: str
    characteristics: str

# AI 모델을 담을 변수를 미리 만들어 둡니다.
# 처음에는 비어있다가, 서버가 켜질 때 채워집니다.
models = {}

# --- 2. 서버가 시작될 때 AI 모델을 로딩하는 함수 ---
# @app.on_event("startup") 데코레이터는 FastAPI 서버가 켜질 때 딱 한 번만 실행됩니다.
# 이렇게 하면 매번 요청이 올 때마다 모델을 로딩하는 비효율을 막을 수 있습니다.
@app.on_event("startup")
def load_models():
    print("AI 모델 로딩을 시작합니다...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 이미지 생성 모델 (Stable Diffusion)
    models["image_pipe"] = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    
    # 텍스트 생성 모델 (Ko-Alpaca)
    models["tokenizer"] = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
    models["text_model"] = AutoModelForCausalLM.from_pretrained(
        "beomi/KoAlpaca-Polyglot-12.8B",
        torch_dtype=torch.float16
    ).to(device)
    
    print("AI 모델 로딩이 완료되었습니다.")

# --- 3. 핵심 API 기능: 프로필 생성 ---
# POST 메소드: 프론트엔드에서 데이터를 '보내서' 무언가를 '만들' 때 사용합니다.
@app.post("/api/v1/generate-profile")
def generate_profile(request: ProfileRequest):
    # 1. 감성적인 소개글 생성
    prompt_text = f"""
    아래 정보를 바탕으로 유기견의 입양을 독려하는 따뜻하고 감성적인 프로필 소개글을 2~3문장으로 작성해줘:
    - 이름: {request.name}
    - 품종: {request.breed}
    - 특징: {request.characteristics}
    ---
    소개글:
    """
    inputs = models["tokenizer"](prompt_text, return_tensors="pt").to(models["text_model"].device)
    output_sequences = models["text_model"].generate(**inputs, max_new_tokens=200)
    generated_text = models["tokenizer"].decode(output_sequences[0], skip_special_tokens=True)
    
    # 2. 멋진 프로필 이미지 생성
    prompt_image = f"A high-resolution, heartwarming studio photo of a cute {request.breed} dog named {request.name}, looking at the camera"
    image = models["image_pipe"](prompt=prompt_image).images[0]
    
    # 3. 생성된 이미지를 웹에서 쓸 수 있도록 Base64로 변환
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 4. 프론트엔드에 결과 반환
    return {
        "profile_text": generated_text.split("소개글:")[1].strip(),
        "profile_image_base64": img_str
    }