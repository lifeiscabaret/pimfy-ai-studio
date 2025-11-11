from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import torch
import numpy as np
import cv2
import io
import base64
import asyncio
import re
import textwrap
import requests
from typing import Optional, List, Tuple, Union

# --- DB ---
import databases
import sqlalchemy

# --- AI 모델 ---
# from diffusers import StableDiffusionXLImg2ImgPipeline # SDXL은 별도 서버로 분리됨
from rembg import new_session, remove
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# GPT-4o 사용
import openai 
import httpx # ⭐️ SDXL 서버 통신을 위한 비동기 HTTP 클라이언트 추가

# --- 1. 환경 변수 및 DB 설정 ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/inday_fileinfo/img")
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "https://www.pimfyvirus.com")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# 'homeprotection' 테이블 정의 (기존 유지)
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

# homeprotectionsub02 테이블 정의 추가
sub02_table = sqlalchemy.Table(
    "homeprotectionsub02",
    metadata,
    sqlalchemy.Column("puid", sqlalchemy.Integer), 
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),
    sqlalchemy.Column("num", sqlalchemy.Integer), 
)


# --- 2. Pydantic 모델 정의 (DB 스키마) ---
class Dog(BaseModel):
    uid: int
    subject: str
    s_pic01: Optional[str] = None
    image_filenames: List[str] = [] 
    
    addinfo03: Optional[str] = None
    addinfo04: Optional[str] = None
    addinfo05: Optional[str] = None
    addinfo07: Optional[str] = None
    addinfo08: Optional[str] = None
    addinfo09: Optional[str] = None
    addinfo10: Optional[str] = None
    addinfo11: Optional[str] = None
    addinfo19: Optional[str] = None

class RealProfileRequest(BaseModel):
    dog_uid: int


# --- 3. FastAPI 앱 & AI 모델 변수 선언 ---
models = {}
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_id = 0 if device == "cuda" else None

# ⭐️ SDXL 서비스 주소 정의
SDXL_SERVICE_URL = "http://sdxl-service:8001/generate/background"


# --- AI 모델 로딩 (서버 시작 시) ---
@app.on_event("startup")
def load_models_and_db():
    print("AI 모델 로딩 시작...")
    print(f"Using device: {device}")
    
    # ⭐️ SDXL 로딩 코드 제거 - VRAM 확보 완료
    # print("Loading Stable Diffusion XL pipeline...") ...

    # (모델 2: GPT-4o API 사용)
    print("KoAlpaca 대신 GPT-4o API를 사용합니다.")
    
    # (모델 3: Real-ESRGAN 로드 - 파일 경로 수정)
    print("Loading Real-ESRGAN model...")
    try:
        model_path = "/app/esrgan/RealESRGAN_x4plus.pth"
        
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        models["upsampler"] = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=esrgan_model,
            # ⭐️ 성능 최적화: Tile Size 복구 (안정화)
            tile=4000, 
            tile_pad=32,
            pre_pad=16,
            half=torch.cuda.is_available(),
            gpu_id=gpu_id
        )
        print("Real-ESRGAN 로드 완료.")
    except Exception as e:
        print(f"🚨 Real-ESRGAN 로드 실패: {e}")

    # (모델 4: rembg 세션 로드)
    print("Loading rembg session...")
    try:
        models["remover"] = new_session(model_name="u2net_human_seg")
        print("rembg 세션 로드 완료.")
    except Exception as e:
        print(f"🚨 rembg 세션 로드 실패: {e}.")

    print("--- 모든 AI 모델 로딩 완료 ---")

# --- DB 연결/해제 ---
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
    
    main_query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog_data = await db.fetch_one(main_query)
    
    if not dog_data:
        raise HTTPException(status_code=404, detail=f"UID {dog_uid}에 해당하는 강아지 정보를 DB에서 찾을 수 없습니다.")

    image_query = sub02_table.select().where(sub02_table.c.puid == dog_uid).order_by(sub02_table.c.num)
    image_data_list = await db.fetch_all(image_query)
    
    image_filenames = [row['s_pic01'] for row in image_data_list]

    return Dog(**dog_data, image_filenames=image_filenames)


# --- 헬퍼 함수 (Image/Text) ---

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


# ⭐️ SDXL 서버 통신 함수 추가
async def call_sdxl_service(base64_dog_image: str, dog_info: dict) -> Image.Image:
    """SDXL 서버에 요청하여 배경 이미지를 생성합니다."""
    
    # ⭐️ SDXL 배경 설정: 파스텔톤과 프롬프트는 추후 DB 필드로 변경 가능
    color_hint = "pastel pink" 
    prompt_detail = f"Minimalist studio background suitable for {dog_info.get('name', 'a dog')}."

    payload = {
        "base64_dog_image": base64_dog_image,
        "prompt": prompt_detail,
        "color_hint": color_hint
    }

    print(f"Calling SDXL service at {SDXL_SERVICE_URL} with color: {color_hint}")
    
    # httpx를 사용하여 비동기적으로 호출
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            response = await client.post(SDXL_SERVICE_URL, json=payload)
            response.raise_for_status()  
            
            result = response.json()
            base64_bg = result.get("base64_background_image")
            
            if not base64_bg:
                raise ValueError("SDXL service returned no background image.")

            # Base64 디코딩하여 PIL Image 객체로 반환
            bg_image_data = base64.b64decode(base64_bg)
            return Image.open(io.BytesIO(bg_image_data)).convert("RGB")

        except httpx.RequestError as e:
            print(f"🚨 SDXL Service Connection/Request Error: {e}")
        except Exception as e:
            print(f"🚨 SDXL Processing Error: {e}")
        
        # 오류 시 또는 서비스 미사용 시 흰색 배경 반환 (안정성 확보)
        print("Returning default white background due to SDXL error.")
        return Image.new('RGB', (800, 1200), (255, 255, 255))
        
# (GPT-4o API를 사용하는 텍스트 생성 - 키-값 리스트 형식 강제 적용)
def generate_dog_text(dog: Dog) -> str:
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    dog_info = f"""
    - 이름: {clean_text(dog.subject)}
    - 성별: {clean_text(dog.addinfo03)}
    - 나이(추정): {clean_text(dog.addinfo05)}
    - 몸무게: {clean_text(dog.addinfo07)}kg
    - 중성화: {clean_text(dog.addinfo04)}
    - 성격 태그: {clean_text(dog.addinfo08)}
    - 성격 및 특징: {clean_text(dog.addinfo10)}
    - 구조 사연: {clean_text(dog.addinfo09)}
    - 병력/건강: {clean_text(dog.addinfo19)}
    - 기타: {clean_text(dog.addinfo11)}
    """

    # 🚨 수정: 출력 형식을 키-값 리스트로 강제
    system_prompt = """
    당신은 유기견의 입양 공고에 사용될 **핵심 정보를 키-값(Key-Value) 쌍의 리스트**로 변환하는 AI 전문가입니다.
    **감정적인 표현은 배제**하고, 요청된 정보를 바탕으로 아래 규칙에 따라 정확하고 간결하게 출력하세요.

    **[생성 규칙]**
    1. 출력은 오직 **항목: 값** 형태의 리스트로만 구성되어야 합니다. (다른 설명 문장 절대 금지)
    2. 모든 항목은 줄바꿈 문자(\n)로 분리되어야 합니다.
    3. 정보가 없는 항목은 출력에서 제외하세요.
    4. 입양 문의 방법은 마지막 줄에 **반드시** '문의: 인스타그램 @lovely4puppies에서 확인하세요.' 형식으로 추가하세요.
    """

    user_content = f"""
    [강아지 정보]:
    {dog_info}

    위 정보를 기반으로 다음 항목들을 Key-Value 형식으로 변환하여 출력해 주세요.
    
    이름: [이름]
    성별: [성별]
    나이: [나이(추정)]
    몸무게: [몸무게]
    중성화: [중성화 여부]
    
    특징: [성격 태그 및 성격/특징 요약]
    건강 상태: [병력/건강 요약]
    사연: [구조 사연 요약]
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    print("GPT-4o 텍스트 생성 시작...")
    
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        generated_text = response.choices[0].message.content.strip()
        
        # 후처리: 문의 항목이 누락될 경우를 대비해 마지막 줄에 명시적으로 추가
        if not generated_text.lower().strip().endswith("확인하세요."):
             generated_text += "\n문의: 인스타그램 @lovely4puppies에서 확인하세요."
             
        print("GPT-4o 텍스트 생성 완료.")
            
    except Exception as e:
        print(f"🚨 GPT-4o API 호출 중 오류 발생: {type(e).__name__}: {e}")
        generated_text = "GPT-4o API 오류로 소개글을 생성할 수 없습니다."
            
    return generated_text

# (이미지 선별 로직 - A-컷 선별 및 누끼 적합성 최적화 적용)
async def select_best_image(dog: Dog) -> Tuple[Union[Image.Image, None], Union[str, None]]:
    best_input_image_pil = None
    best_score = -1 
    original_rgb_image_base64 = None

    image_filenames = []
    if dog.s_pic01:
        image_filenames.append(dog.s_pic01)
    image_filenames.extend(dog.image_filenames)

    if not image_filenames:
        print(f"[{dog.uid}] !! 유효한 이미지 파일이 없습니다.")
        return None, None
    
    remover_session = models.get("remover")
    if not remover_session:
        raise RuntimeError("🚨 rembg 세션이 로드되지 않았습니다.")

    print(f"[{dog.uid}] 최적 이미지 선별 시작...")
    
    MAX_FOCUS_SCORE = 4000 # 기준값 상향 (클로즈업 Focus 점수 희석)

    for filename in image_filenames:
        if not filename or filename.strip() == "":
            continue
        try:
            image_url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{filename}"
            
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()
            
            input_image_pil = Image.open(response.raw).convert("RGB")
            
            # 1. 선명도(Focus) 측정
            cv2_gray = cv2.cvtColor(pil_to_cv2(input_image_pil), cv2.COLOR_BGR2GRAY)
            focus_measure = cv2.Laplacian(cv2_gray, cv2.CV_64F).var()
            
            # 2. rembg를 사용하여 마스크 크기 측정 (강아지 크기)
            removed_bg_image = remove(
                input_image_pil, 
                session=remover_session, 
                alpha_matting=True 
            )
            alpha_mask = np.array(removed_bg_image.split()[3])
            mask_size = np.count_nonzero(alpha_mask > 10)
            
            # 3. 종합 점수 계산 (누끼 적합성 최우선)
            
            # ⭐️ 이미지 선별 최적화: 마스크 크기 정규화 기준을 동적으로 설정 (전체 픽셀의 20%를 최적으로)
            width, height = input_image_pil.size
            aspect_ratio = max(width, height) / min(width, height)
            
            # 구도 보너스: 세로(portrait) 구도일 때 (+0.2 보너스)
            orientation_bonus = 0.0
            if aspect_ratio < 1.1 or (height > width and aspect_ratio < 1.5):
                orientation_bonus = 0.2 
            
            # 3. 종합 점수 계산 (누끼 적합성 및 구도 100% 반영)
            TARGET_MASK_SIZE = (width * height) * 0.20 # 강아지가 화면의 20% 차지할 때 최대 점수
            
            # ⭐️ Mask Size 초과 시 패널티 적용
            if mask_size > TARGET_MASK_SIZE:
                # 20% 초과 시 초과 점수 획득을 막고 1.0으로 고정 (클로즈업 사진 방지)
                normalized_mask = 1.0
            else:
                # 20% 이하일 때는 비율대로 점수를 부여
                normalized_mask = mask_size / TARGET_MASK_SIZE
            
            normalized_focus = min(focus_measure, MAX_FOCUS_SCORE) / MAX_FOCUS_SCORE
            
            # ⭐️ 최종 가중치: Mask Ratio 100% + 구도 보너스 (Focus Score 기여도 0)
            composite_score = (normalized_mask * 1.0) + orientation_bonus + (normalized_focus * 0.0)
            
            if composite_score > best_score:
                print(f"     >>> ★★★ 새 최적 이미지 발견! (점수: {composite_score:.4f}, 마스크: {mask_size}, 선명도: {focus_measure:.2f}, 파일: {filename})")
                best_score = composite_score 
                best_input_image_pil = input_image_pil
                
                buffered_original = io.BytesIO()
                best_input_image_pil.save(buffered_original, format="PNG")
                original_rgb_image_base64 = base64.b64encode(buffered_original.getvalue()).decode("utf-8")
                
        except requests.exceptions.HTTPError as e:
            print(f"     ! 이미지 처리 중 오류 (HTTP 오류 - 404 등): {image_url} / 오류: {e}")
            continue
        except Exception as e:
            print(f"     ! 이미지 처리 중 오류 (PIL/기타 오류 - 파일 식별 실패 등): {image_url} / 오류: {e}")
            continue
            
    return best_input_image_pil, original_rgb_image_base64


# --- API 엔드포인트 ---

@app.post("/api/v1/generate-real-profile", response_model=dict)
async def generate_real_profile(request: RealProfileRequest):
    if "upsampler" not in models or "remover" not in models:
        raise HTTPException(status_code=503, detail="AI 모델(Upsampler/Remover)이 로드되지 않았습니다.")

    dog = await get_dog_details(request.dog_uid)
    
    # 1. 최적의 이미지 선별
    best_input_image_pil, original_rgb_image_base64 = await select_best_image(dog)
    
    final_image_base64 = ""
    generated_text = ""

    try:
        if best_input_image_pil:
            # 2. Real-ESRGAN으로 화질 복원 
            cv2_image = pil_to_cv2(best_input_image_pil)
            upscaled_image_cv2, _ = models["upsampler"].enhance(cv2_image, outscale=4)
            upscaled_image_pil = cv2_to_pil(upscaled_image_cv2)
            print("화질 복원 완료.")

            # 3. 복원된 이미지의 배경 제거 (rembg)
            print("배경 제거(rembg) 시작...")
            remover_session = models.get("remover") 
            removed_bg_image = remove(
                upscaled_image_pil,
                session=remover_session,
                alpha_matting=True
            )
            print("배경 제거 완료.")

            # ⭐️ 4. SDXL 서버 호출 및 배경 이미지 받기 (마이크로서비스 연동)
            dog_info_dict = {"name": dog.subject}
            
            # 누끼 딴 이미지를 Base64 (PNG)로 인코딩
            temp_buffer = io.BytesIO()
            removed_bg_image.save(temp_buffer, format="PNG") 
            base64_dog_image_png = base64.b64encode(temp_buffer.getvalue()).decode("utf-8")

            # SDXL 서버 호출
            sdxl_bg_image_pil = await call_sdxl_service(base64_dog_image_png, dog_info_dict)
            
            # 5. 텍스트 생성 (GPT-4o)
            generated_text = generate_dog_text(dog)
            
            # 6. Pillow 템플릿 합성 (SDXL 배경 사용)
            print("Pillow 템플릿 합성 시작...")
            template_width = 800
            template_height = 1200
            
            # ⭐️ SDXL 배경 이미지를 템플릿으로 사용
            template = sdxl_bg_image_pil.resize((template_width, template_height))
            # draw는 RGB 이미지에만 사용 가능하므로, 배경 이미지에서 draw 객체 생성
            draw = ImageDraw.Draw(template) 

            try:
                font_title = ImageFont.truetype("/app/NanumGothic-Bold.ttf", 40)
                font_body = ImageFont.truetype("/app/NanumGothic-Regular.ttf", 24)
            except IOError:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
                print("!! 폰트 파일 로드 실패. 기본 폰트 사용.")


            img_height = int(template_width * (removed_bg_image.height / removed_bg_image.width))
            
            # ⭐️ 배경 위에 누끼 딴 강아지 이미지(RGBA) 합성
            image_to_template = removed_bg_image.resize((template_width, img_height))
            template.paste(image_to_template, (0, 0), image_to_template) 

            # 텍스트 출력
            text_y_position = img_height + 30
            draw.text((30, text_y_position), dog.subject, font=font_title, fill=(0,0,0))
            text_y_position += 60

            lines = generated_text.split('\n')
            for line in lines:
                draw.text((30, text_y_position), line.strip(), font=font_body, fill=(50, 50, 50))
                text_y_position += 30

            buffered = io.BytesIO()
            template.save(buffered, format="PNG")
            final_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print("Pillow 템플릿 합성 완료.")

        else:
            print(f"[{dog.uid}] !! 치명적 오류: 유효한 이미지가 없어 프로필 생성을 중단합니다.")
            generated_text = "프로필을 생성할 수 없습니다: 유효한 원본 이미지가 없습니다."
            final_image_base64 = ""
            
    except Exception as e:
        print(f"[{dog.uid}] !! 이미지/텍스트 처리 중 오류: {e}. 원본 이미지를 반환합니다.")
        final_image_base64 = original_rgb_image_base64 or "Error: Template composition failed."
        generated_text = generated_text or "프로필 생성 중 오류가 발생했습니다."
    
    # ⭐️ 성능 최적화 3: 요청 종료 후 GPU 메모리 정리
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()
        
    return {
        "profile_text": generated_text,
        "profile_image_base64": final_image_base64
    }

@app.get("/api/dogs/{dog_uid}", response_model=Dog)
async def get_dog_details_api(dog_uid: int):
    return await get_dog_details(dog_uid)
