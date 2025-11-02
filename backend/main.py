from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import torch
import numpy as np
import cv2 # (추가!) Real-ESRGAN이 사용
from io import BytesIO
import base64
import asyncio
import re
import textwrap

# --- AI 모델 ---
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from rembg import remove

# --- (✨ 추가!) 화질 복원 (Real-ESRGAN) ---
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- (✨ 추가!) Pillow 효과 ---
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- DB ---
import databases
import sqlalchemy
import requests

# --- 1. 환경 변수 및 DB 설정 ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/inday_fileinfo/img")
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "https://www.pimfyvirus.com")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# 'homeprotection' 테이블 정의 (컬럼명 수정됨)
dogs_table = sqlalchemy.Table(
    "homeprotection",
    metadata,
    sqlalchemy.Column("uid", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("subject", sqlalchemy.String(250)),
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),
    sqlalchemy.Column("addinfo01", sqlalchemy.String(100)),
    sqlalchemy.Column("addinfo02", sqlalchemy.String(100)),
    sqlalchemy.Column("addinfo12", sqlalchemy.String(250)),
    sqlalchemy.Column("addinfo15", sqlalchemy.String(250)),
    sqlalchemy.Column("addinfo03", sqlalchemy.String(10)),
    sqlalchemy.Column("addinfo04", sqlalchemy.String(10)),
    sqlalchemy.Column("addinfo05", sqlalchemy.String(10)),
    sqlalchemy.Column("addinfo07", sqlalchemy.String(10)),
    sqlalchemy.Column("addinfo08", sqlalchemy.Text),
    sqlalchemy.Column("addinfo09", sqlalchemy.Text),
    sqlalchemy.Column("addinfo10", sqlalchemy.Text),
    sqlalchemy.Column("addinfo11", sqlalchemy.Text),
    sqlalchemy.Column("addinfo19", sqlalchemy.String(250)),
)

# --- 2. Pydantic 모델 정의 ---
class Dog(BaseModel):
    uid: int
    subject: str
    s_pic01: str | None
    addinfo01: str | None
    addinfo02: str | None
    addinfo12: str | None
    addinfo15: str | None
    addinfo03: str | None
    addinfo04: str | None
    addinfo05: str | None
    addinfo07: str | None
    addinfo08: str | None
    addinfo09: str | None
    addinfo10: str | None
    addinfo11: str | None
    addinfo19: str | None

class RealProfileRequest(BaseModel):
    dog_uid: int

# (✨ 추가!) 마케팅 이미지 요청 모델
class MarketingProfileRequest(BaseModel):
    dog_uid: int
    creative_prompt: str # "꽃밭에서 웃고 있는", "크리스마스 스웨터를 입은" 등

class ProfileResponse(BaseModel):
    profile_text: str
    profile_image_base64: str

# --- 3. FastAPI 앱 & AI 모델 변수 선언 ---
models = {}
app = FastAPI()

# --- 4. AI 모델 로딩 (서버 시작 시) ---
@app.on_event("startup")
def load_models_and_db():
    print("AI 모델 로딩을 시작합니다... (SDXL, KoAlpaca, Real-ESRGAN)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # (✨ 수정!) Real-ESRGAN GPU ID 설정
    gpu_id = 0 if device == "cuda" else None

    # (모델 1: SDXL 로드 - 마케팅 API용)
    print("Loading Stable Diffusion XL (SDXL) pipeline...")
    models["image_pipe"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    print("SDXL 로드 완료.")

    # (모델 2: KoAlpaca 로드 - 텍스트 생성용)
    print("Loading KoAlpaca-Polyglot-5.8B model...")
    models["tokenizer"] = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    models["text_model"] = AutoModelForCausalLM.from_pretrained(
        "beomi/KoAlpaca-Polyglot-5.8B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    print("KoAlpaca 로드 완료.")
    
    # (✨ 추가! 모델 3: Real-ESRGAN 로드 - 화질 복원용)
    print("Loading Real-ESRGAN model...")
    try:
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        models["upsampler"] = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            dni_weight=None,
            model=esrgan_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(), # FP16 사용
            gpu_id=gpu_id
        )
        print("Real-ESRGAN 로드 완료.")
    except Exception as e:
        print(f"🚨 Real-ESRGAN 로드 실패: {e}. /generate-real-profile API가 작동하지 않을 수 있습니다.")

    print("--- 모든 AI 모델 로딩 완료 ---")

# --- 5. DB 연결/해제 및 헬퍼 함수 ---
@app.on_event("shutdown")
async def shutdown_db_client():
    if database.is_connected:
        await database.disconnect()

async def get_db_connection():
    if not database.is_connected:
        await database.connect()
    return database

async def get_dog_details(dog_uid: int) -> Dog:
    db = await get_db_connection()
    query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog_data = await db.fetch_one(query)
    if not dog_data:
        raise HTTPException(status_code=404, detail=f"UID {dog_uid}에 해당하는 강아지 정보를 DB에서 찾을 수 없습니다.")
    return Dog(**dog_data)

# (✨ 추가!) PIL <-> CV2 변환 헬퍼
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# (✨ 추가!) 텍스트 생성 헬퍼 (중복 제거)
def generate_dog_text(dog: Dog, tokenizer, model):
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    dog_subject = clean_text(dog.subject)
    dog_gender = clean_text(dog.addinfo03)
    dog_birth = clean_text(dog.addinfo05)
    dog_weight = clean_text(dog.addinfo07)
    dog_neuter = clean_text(dog.addinfo04)
    dog_tags = clean_text(dog.addinfo08)
    dog_personality = clean_text(dog.addinfo10)
    dog_story = clean_text(dog.addinfo09)
    dog_illness = clean_text(dog.addinfo19)
    dog_etc = clean_text(dog.addinfo11)

    prompt_text = f"""
# MISSION (임무)
당신은 유기동물 입양 홍보 전문 카피라이터입니다. [견종 정보]만을 이용해서, 이 아이의 매력과 사연이 잘 드러나는 감성적인 입양 프로필 소개글을 작성해야 합니다.

# INSTRUCTIONS (작성 지침)
1.  **임무:** 당신은 [견종 정보]를 바탕으로, 따뜻하고 긍정적인 '입양 홍보 문구'를 작성합니다.
2.  **재각색:** [견종 정보]의 **내용을 바탕으로** 하되, '그대로 복사하지 말고' **부드러운 문장으로 재각색**합니다.
3.  **환각 금지:** [견종 정보]에 **없는 내용(흡연, 소득, 혈액형 등 사람 정보)은 절대 지어내지 마세요.**
4.  **분량:** 2~3 문단으로 짧게 작성하세요.

# 견종 정보 (Dog's Data)
- 이름: {dog_subject}
- 성별: {dog_gender}
- 나이(추정): {dog_birth}
- 몸무게: {dog_weight}kg
- 중성화: {dog_neuter}
- 성격 태그: {dog_tags}
- 성격 및 특징: {dog_personality}
- 구조 사연: {dog_story}
- 병력/건강: {dog_illness}
- 기타: {dog_etc}
---
# PROFILE (프로필 작성)
소개글:
"""
    
    print("KoAlpaca 텍스트 생성 시작...")
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            temperature=0.2, # 0.2로 낮춰서 환각 억제
            repetition_penalty=1.2,
            early_stopping=True
        )
        decoded_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = decoded_text.split("소개글:")[-1].strip()
        print("텍스트 생성 완료.")
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        generated_text = "소개글을 생성하는 중 오류가 발생했습니다."
    return generated_text

# (✨ 추가!) 최적 이미지 선별 헬퍼 (중복 제거)
async def select_best_image(dog: Dog) -> (Image.Image | None, str | None):
    best_input_image_pil = None
    best_mask_size = 0
    original_rgb_image_base64 = None

    image_filenames = [
        dog.s_pic01, dog.addinfo01, dog.addinfo02, 
        dog.addinfo12, dog.addinfo15
    ]
    
    print(f"[{dog.uid}] 최적 이미지 선별 시작...")
    for filename in image_filenames:
        if not filename or filename.strip() == "":
            continue
        try:
            image_url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{filename}"
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()
            input_image_pil = Image.open(response.raw).convert("RGB")
            
            removed_bg_image = remove(input_image_pil, alpha_matting=True)
            alpha_mask = np.array(removed_bg_image.split()[3])
            mask_size = np.count_nonzero(alpha_mask > 10)
            
            if mask_size > best_mask_size:
                print(f"    >>> ★★★ 새 최적 이미지 발견! (마스크 크기: {mask_size})")
                best_mask_size = mask_size
                best_input_image_pil = input_image_pil
                buffered_original = BytesIO()
                best_input_image_pil.save(buffered_original, format="PNG")
                original_rgb_image_base64 = base64.b64encode(buffered_original.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"    ! 이미지 처리 중 오류 (무시): {e}")
            continue
    return best_input_image_pil, original_rgb_image_base64

# --- 6. API 엔드포인트 ---

# --- (✨ API 1: 공식 프로필 생성 - Real-ESRGAN 사용) ---
@app.post("/api/v1/generate-real-profile", response_model=ProfileResponse)
async def generate_real_profile(request: RealProfileRequest):
    if "upsampler" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델(Upsampler 또는 Text)이 로드되지 않았습니다.")

    dog = await get_dog_details(request.dog_uid)
    
    # 1. 최적의 이미지 선별
    best_input_image_pil, original_rgb_image_base64 = await select_best_image(dog)
    
    image_to_template = None
    final_image_base64 = "Error: Image generation failed."

    if best_input_image_pil:
        try:
            # 2. (✨) Real-ESRGAN으로 화질 복원 (PIL -> CV2 -> Enhance -> PIL)
            print(f"[{dog.uid}] Real-ESRGAN 화질 복원 시작...")
            cv2_image = pil_to_cv2(best_input_image_pil)
            upscaled_image_cv2, _ = models["upsampler"].enhance(cv2_image, outscale=4)
            upscaled_image_pil = cv2_to_pil(upscaled_image_cv2)
            print("화질 복원 완료.")

            # 3. (✨) 복원된 이미지의 배경 제거
            print("배경 제거(rembg) 시작...")
            removed_bg_image = remove(upscaled_image_pil, alpha_matting=True) # RGBA
            print("배경 제거 완료.")

            # 4. (✨) Pillow로 가장자리 블러 처리
            print("가장자리 블러 처리 (GaussianBlur) 시작...")
            alpha = removed_bg_image.split()[3]
            blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=5)) # 5px 블러
            removed_bg_image.putalpha(blurred_alpha)
            print("가장자리 블러 처리 완료.")
            
            image_to_template = removed_bg_image # (템플릿에 사용할 최종 이미지)

        except Exception as e:
            print(f"[{dog.uid}] !! 이미지 복원/처리 중 오류: {e}")
            # (복원 실패 시) 원본이라도 사용
            if best_input_image_pil:
                print("원본 이미지를 대신 사용합니다.")
                image_to_template = best_input_image_pil.convert("RGBA") # 템플릿용으로 RGBA 변환
            
    # 5. 텍스트 생성
    generated_text = generate_dog_text(dog, models["tokenizer"], models["text_model"])

    # 6. (✨) Pillow 템플릿 합성
    if image_to_template:
        try:
            print("Pillow 템플릿 합성 시작...")
            template_width = 800
            template_height = 1200
            template = Image.new('RGB', (template_width, template_height), (255, 255, 255))
            draw = ImageDraw.Draw(template)

            font_title = ImageFont.truetype("/app/NanumGothic-Bold.ttf", 40)
            font_body = ImageFont.truetype("/app/NanumGothic-Regular.ttf", 24)

            # (✨) 리사이즈 및 RGBA 마스크를 사용한 붙여넣기
            img_height = int(template_width * (image_to_template.height / image_to_template.width))
            image_to_template = image_to_template.resize((template_width, img_height))
            
            # (✨) RGBA의 투명/블러 영역을 살려서 붙여넣기
            template.paste(image_to_template, (0, 0), image_to_template) 

            text_y_position = img_height + 30
            draw.text((30, text_y_position), dog.subject, font=font_title, fill=(0,0,0))
            text_y_position += 60

            lines = textwrap.wrap(generated_text, width=60)
            for line in lines:
                draw.text((30, text_y_position), line, font=font_body, fill=(50, 50, 50))
                text_y_position += 30

            buffered = BytesIO()
            template.save(buffered, format="PNG")
            final_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print("Pillow 템플릿 합성 완료.")

        except IOError as e:
            print(f"!! 폰트 파일 로드 실패: {e}. Dockerfile 확인!")
            final_image_base64 = original_rgb_image_base64 or "Error: Font file missing."
        except Exception as e:
            print(f"[{dog.uid}] !! 템플릿 합성 중 오류: {e}")
            final_image_base64 = original_rgb_image_base64 or "Error: Template composition failed."
            
    elif original_rgb_image_base64:
        # (이미지 처리는 실패했지만 원본은 있을 경우)
        final_image_base64 = original_rgb_image_base64
    else:
        # (유효한 이미지가 아예 없는 경우)
        print(f"[{dog.uid}] !! 치명적 오류: 유효한 이미지가 없어 프로필 생성을 중단합니다.")
        generated_text = "프로필을 생성할 수 없습니다: 유효한 원본 이미지가 없습니다."
        final_image_base64 = ""
        
    return {
        "profile_text": generated_text,
        "profile_image_base64": final_image_base64
    }

# --- (✨ API 2: 마케팅 이미지 생성 - SDXL 사용) ---
@app.post("/api/v1/generate-marketing-image", response_model=ProfileResponse)
async def generate_marketing_image(request: MarketingProfileRequest):
    if "image_pipe" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델(SDXL 또는 Text)이 로드되지 않았습니다.")

    dog = await get_dog_details(request.dog_uid)

    # 1. 최적의 이미지 선별
    best_input_image_pil, original_rgb_image_base64 = await select_best_image(dog)
    
    final_image_base64 = "Error: SDXL Image generation failed."

    if best_input_image_pil:
        try:
            print(f"[{dog.uid}] SDXL 마케팅 이미지 생성 시작...")
            # 2. SDXL 입력용 전처리
            output_image = remove(best_input_image_pil, alpha_matting=True)
            output_image = output_image.resize((1024, 1024))
            rgb_image_for_sd = Image.new("RGB", (1024, 1024), (255, 255, 255))
            rgb_image_for_sd.paste(output_image, mask=output_image.split()[3])

            dog_name = dog.subject if dog.subject else "this dog"
            
            # 3. (✨) 사용자 프롬프트와 기본 프롬프트 결합
            prompt_image = f"""
            (masterpiece, best quality, high resolution, photo-realistic:1.2),
            {request.creative_prompt},
            (professional studio portrait photo of {dog_name}),
            sharp focus, highly detailed fur texture, natural lighting
            """.strip().replace("\n", " ")
            
            negative_prompt = "blurry, low quality, worst quality, cartoon, drawing, sketch, illustration, anime, 3d render, watermark, text"

            print(f"Using SDXL Prompt: {prompt_image}")
            
            # 4. SDXL 생성
            enhanced_image = models["image_pipe"](
                prompt=prompt_image,
                negative_prompt=negative_prompt,
                image=rgb_image_for_sd,
                strength=0.65, # (창의성을 위해 strength를 조금 높게 설정, 0.6~0.75 테스트)
                guidance_scale=8.0
            ).images[0]
            print("SDXL 이미지 개선 완료.")

            buffered = BytesIO()
            enhanced_image.save(buffered, format="PNG")
            final_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"[{dog.uid}] !! SDXL 생성 중 오류: {e}")
            final_image_base64 = original_rgb_image_base64 or "Error: SDXL failed."

    # 5. 텍스트 생성 (동일한 텍스트 로직 재사용)
    generated_text = generate_dog_text(dog, models["tokenizer"], models["text_model"])

    if not best_input_image_pil:
        final_image_base64 = "Error: No valid source image."
        generated_text = "프로필을 생성할 수 없습니다: 유효한 원본 이미지가 없습니다."

    return {
        "profile_text": generated_text,
        "profile_image_base64": final_image_base64
    }

# --- (기존 API: /api/dogs, /api/dogs/{dog_uid} - 변경 없음) ---
@app.get("/api/dogs", response_model=list[Dog])
async def get_dog_list(search: str | None = None):
    db = await get_db_connection()
    query = dogs_table.select()
    if search:
        query = query.where(
            (dogs_table.c.subject.ilike(f"%{search}%")) |
            (dogs_table.c.addinfo10.ilike(f"%{search}%"))
        )
    results = await db.fetch_all(query)
    return [dict(row) for row in results]

@app.get("/api/dogs/{dog_uid}", response_model=Dog)
async def get_dog_details_api(dog_uid: int):
    db = await get_db_connection()
    query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog = await db.fetch_one(query)
    if not dog:
        raise HTTPException(status_code=404, detail="해당 ID의 강아지 정보를 찾을 수 없습니다.")
    return dict(dog)
