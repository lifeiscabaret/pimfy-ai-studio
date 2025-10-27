# (터미널에 "cat > main.py" 입력 후, 이 코드를 붙여넣고 Enter, Ctrl+D 를 누르세요)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import torch
# --- (수정!) ---
from diffusers import StableDiffusionXLImg2ImgPipeline # (SD 1.5 -> SDXL)
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from io import BytesIO
import asyncio
import databases
import sqlalchemy
import requests
import numpy as np # (추가!)
from PIL import Image, ImageDraw, ImageFont # (추가!)
from rembg import remove
import textwrap # (추가!)
import re # (추가!)

# --- 1. Cafe24 DB (MySQL) 설정 ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/inday_fileinfo/img")
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "https://www.pimfyvirus.com") # .env에서 읽거나 기본값 사용

if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# 'homeprotection' 테이블 정의
dogs_table = sqlalchemy.Table(
    "homeprotection",
    metadata,
    sqlalchemy.Column("uid", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("subject", sqlalchemy.String(250)),
    # --- (수정!) s_pic01 및 add... 컬럼 ---
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),
    sqlalchemy.Column("add011", sqlalchemy.String(150)),
    sqlalchemy.Column("add012", sqlalchemy.String(150)),
    sqlalchemy.Column("add013", sqlalchemy.String(150)),
    sqlalchemy.Column("add014", sqlalchemy.String(150)),
    sqlalchemy.Column("add015", sqlalchemy.String(150)),
    # -----------------------------------
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
    # --- (수정!) s_pic01 및 add... 컬럼 (None 허용) ---
    s_pic01: str | None
    add011: str | None
    add012: str | None
    add013: str | None
    add014: str | None
    add015: str | None
    # ----------------------------------------------
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

class ProfileResponse(BaseModel):
    profile_text: str
    profile_image_base64: str # (최종 포스터 이미지)

# --- 3. FastAPI 앱 & AI 모델 변수 선언 ---
models = {}
app = FastAPI()

# --- 4. AI 모델 로딩 (서버 시작 시) ---
@app.on_event("startup")
def load_models_and_db():
    print("Cafe24 DB 연결 준비... (각 API 요청 시 연결)")

    print("AI 모델 로딩을 시작합니다... (시간이 몇 분 정도 걸릴 수 있습니다)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- (Step 1: SDXL로 업그레이드) ---
    print("Loading Stable Diffusion XL (SDXL) Image-to-Image pipeline...")
    models["image_pipe"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    # --------------------------------

    print("Loading KoAlpaca-Polyglot-5.8B model...")
    models["tokenizer"] = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    models["text_model"] = AutoModelForCausalLM.from_pretrained(
        "beomi/KoAlpaca-Polyglot-5.8B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    print("AI 모델 로딩이 완료되었습니다.")


# --- 5. DB 자동 연결/해제 이벤트 ---
@app.on_event("shutdown")
async def shutdown_db_client():
    if database.is_connected:
        await database.disconnect()
        print("Cafe24 DB 연결이 해제되었습니다.")

async def get_db_connection():
    if not database.is_connected:
        await database.connect()
    return database

# --- 6. 핵심 기능 API (DB 연동) ---
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
async def get_dog_details(dog_uid: int):
    db = await get_db_connection()
    query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog = await db.fetch_one(query)
    if not dog:
        raise HTTPException(status_code=404, detail="해당 ID의 강아지 정보를 찾을 수 없습니다.")
    return dict(dog)


# --- 7. AI 프로필 생성 API (Step 1, 2, 3 모두 적용) ---
@app.post("/api/v1/generate-real-profile", response_model=ProfileResponse)
async def generate_real_profile(request: RealProfileRequest):
    if "image_pipe" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았거나 준비 중입니다.")

    try:
        dog_dict = await get_dog_details(request.dog_uid)
    except HTTPException as e:
        if e.status_code == 404:
             raise HTTPException(status_code=404, detail=f"UID {request.dog_uid}에 해당하는 강아지 정보를 DB에서 찾을 수 없습니다.")
        else:
             raise e
             
    dog = Dog(**dog_dict)

    # --- (Step 2: 최적 이미지 선별 로직) ---
    best_input_image_pil = None
    best_mask_size = 0
    original_rgb_image_base64 = None 

    # --- (수정!) 실제 add... 컬럼명으로 리스트 생성 ---
    image_filenames = [
        dog.s_pic01, 
        dog.add011, 
        dog.add012, 
        dog.add013, 
        dog.add014,
        dog.add015
    ]
    # ---------------------------------------------
    
    print(f"[{dog.uid}] 최적 이미지 선별 시작. (후보: {len(image_filenames)}개)")

    for filename in image_filenames:
        if not filename or filename.strip() == "": # 비어있는 컬럼 스킵
            continue

        try:
            image_url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{filename}"
            print(f"  - 후보 다운로드 시도: {image_url}")
            
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()

            input_image_pil = Image.open(response.raw).convert("RGB")
            
            print(f"    ... 배경 제거(rembg) 시도...")
            removed_bg_image = remove(input_image_pil, alpha_matting=True) 
            
            alpha_mask = np.array(removed_bg_image.split()[3])
            mask_size = np.count_nonzero(alpha_mask > 10)
            
            print(f"    ... 마스크 크기: {mask_size}")

            if mask_size > best_mask_size:
                print(f"    >>> ★★★ 새 최적 이미지 발견! (이전: {best_mask_size})")
                best_mask_size = mask_size
                best_input_image_pil = input_image_pil
                
                buffered_original = BytesIO()
                best_input_image_pil.save(buffered_original, format="PNG")
                original_rgb_image_base64 = base64.b64encode(buffered_original.getvalue()).decode("utf-8")

        except Exception as e:
            print(f"    ! 이미지 처리 중 오류 (무시하고 다음 후보로): {e}")
            continue
    
    # --- 이미지 처리 (AI 생성 및 템플릿 합성) ---
    img_str = "Error: Image generation failed."
    enhanced_image = None
    generated_text = "Error generating text" # 텍스트 기본값

    if best_input_image_pil:
        print(f"[{dog.uid}] 최종 이미지 선택 완료. AI 생성 시작 (SDXL)...")
        try:
            output_image = remove(best_input_image_pil, alpha_matting=True)
            output_image = output_image.resize((1024, 1024))
            
            rgb_image_for_sd = Image.new("RGB", (1024, 1024), (255, 255, 255))
            rgb_image_for_sd.paste(output_image, mask=output_image.split()[3])

            dog_name = dog.subject if dog.subject else "this dog"
            personality_tags = f", {dog.addinfo08}" if dog.addinfo08 else ""
            
            prompt_image = f"professional studio portrait photo of {dog_name}{personality_tags}, centered, medium shot view, high resolution, masterpiece, best quality, sharp focus, highly detailed fur texture, natural lighting, photo-realistic, cinematic quality, plain light gray background".strip().replace("\n", " ")
            negative_prompt = "blurry, low quality, worst quality, unclear, unfocused, distorted, disfigured, ugly, deformed, bad anatomy, extra limbs, missing limbs, mutated hands, mutation, cartoon, drawing, sketch, illustration, painting, anime, 3d render, illustration, drawing, painting, sketch, cartoon, anime, manga, doll, toy, plastic, fake, watermark, text, signature, words, letters, noise, grain, artifacts, compression artifacts, jpeg artifacts, overexposed, underexposed, bad lighting, multiple dogs, human, person, hands, feet, cage, bars, leash, harness, chain, wires, fence, outdoor, nature, grass, trees, buildings, furniture, messy background, cluttered background".strip().replace("\n", " ")

            print("SDXL 이미지 개선 시작...")
            enhanced_image = models["image_pipe"](
                prompt=prompt_image,
                negative_prompt=negative_prompt,
                image=rgb_image_for_sd,
                strength=0.5,
                guidance_scale=8.0
            ).images[0]
            print("SDXL 이미지 개선 완료.")

        except Exception as e:
            print(f"[{dog.uid}] !! AI 이미지 생성 중 심각한 오류 발생: {e}")
            
    # --- 텍스트 생성 ---
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
1.  **[견종 정보] 활용:** 반드시 아래 제공된 [견종 정보] 내용만을 바탕으로 작성하세요. **정보에 없는 내용은 절대 지어내지 마세요.**
2.  **긍정적 관점:** 아픈 사연이나 건강 문제는 '극복'과 '희망'의 관점에서 긍정적으로 표현해주세요. 특징은 매력으로 강조해주세요.
3.  **감성적 호소:** 입양 희망자가 이 아이와 함께하는 미래를 꿈꾸게 만들고, 아이에게 깊은 애정을 느끼도록 감성적으로 작성해주세요.
4.  **관찰자 시점:** 따뜻하고 다정한 3인칭 관찰자 시점으로 작성해주세요. (예: "밤이는 애교가 많아요.")
5.  **분량:** 2~3 문단 정도의 너무 길지 않은 소개글로 작성해주세요.

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
        inputs = models["tokenizer"](prompt_text, return_tensors="pt").to(models["text_model"].device)
        output_sequences = models["text_model"].generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            temperature=0.6,
            repetition_penalty=1.2,
            early_stopping=True
        )
        decoded_text = models["tokenizer"].decode(output_sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = decoded_text.split("소개글:")[-1].strip()
        print("텍스트 생성 완료.")
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        generated_text = "소개글을 생성하는 중 오류가 발생했습니다."


    # --- (Step 3: Pillow 템플릿 합성) ---
    final_image_for_template = None
    if enhanced_image:
        final_image_for_template = enhanced_image
    elif best_input_image_pil:
        print("AI 생성 실패. 원본 이미지로 템플릿을 생성합니다.")
        final_image_for_template = best_input_image_pil
    
    if final_image_for_template:
        try:
            print("Pillow 템플릿 합성 시작...")
            template_width = 800
            template_height = 1200
            template = Image.new('RGB', (template_width, template_height), (255, 255, 255))
            draw = ImageDraw.Draw(template)

            try:
                # (주의!) Dockerfile에 /app/NanumGothicBold.ttf, /app/NanumGothic.ttf 가 ADD 되어있어야 함
                font_title = ImageFont.truetype("/app/NanumGothic-Bold.ttf", 40)
                font_body = ImageFont.truetype("/app/NanumGothic-Regular.ttf", 24)
            except IOError:
                print("! 경고: 폰트 파일(/app/NanumGothic*.ttf) 로드 실패. Dockerfile 확인! 기본 폰트 사용.")
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()

            img_height = int(template_width * (final_image_for_template.height / final_image_for_template.width))
            final_image_for_template = final_image_for_template.resize((template_width, img_height))
            template.paste(final_image_for_template, (0, 0))

            text_y_position = img_height + 30
            
            draw.text((30, text_y_position), dog.subject, font=font_title, fill=(0,0,0))
            text_y_position += 60

            lines = textwrap.wrap(generated_text, width=60) # 너비(글자 수)는 템플릿에 맞게 조절 필요
            
            for line in lines:
                draw.text((30, text_y_position), line, font=font_body, fill=(50, 50, 50))
                text_y_position += 30 # 줄 간격

            buffered = BytesIO()
            template.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print("Pillow 템플릿 합성 완료.")

        except Exception as e:
            print(f"[{dog.uid}] !! 템플릿 합성 중 오류 발생: {e}")
            if original_rgb_image_base64: # 합성 실패 시 원본이라도 반환
                img_str = original_rgb_image_base64
            else:
                img_str = "Error: Template composition failed."
                
    else:
        print(f"[{dog.uid}] !! 치명적 오류: 유효한 이미지가 없어 프로필 생성을 중단합니다.")
        generated_text = "프로필을 생성할 수 없습니다: 유효한 원본 이미지가 없습니다."
        img_str = "" # 빈 이미지 반환
        
    return {
        "profile_text": generated_text,
        "profile_image_base64": img_str
    }