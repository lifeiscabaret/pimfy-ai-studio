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
from diffusers import StableDiffusionXLImg2ImgPipeline
from rembg import new_session, remove
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# (✨ v30 적용) GPT-4o 사용
import openai # openai>=1.0.0 버전 클라이언트 사용

# --- 1. 환경 변수 및 DB 설정 ---
load_dotenv()
# (✨ DB URL은 .env 파일에서 가져옵니다)
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
    sqlalchemy.Column("addinfo01", sqlalchemy.String(100)), # 레거시 텍스트 필드는 구조 유지를 위해 정의만 남김
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
    sqlalchemy.Column("puid", sqlalchemy.Integer), # homeprotection.uid와 연결됨
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),
    sqlalchemy.Column("num", sqlalchemy.Integer), 
)


# --- 2. Pydantic 모델 정의 (DB 스키마) ---
class Dog(BaseModel):
    uid: int
    subject: str
    s_pic01: Optional[str] = None
    # 새로운 이미지 파일명 리스트 추가
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

# --- AI 모델 로딩 (서버 시작 시) ---
@app.on_event("startup")
def load_models_and_db():
    print("AI 모델 로딩 시작...")
    print(f"Using device: {device}")
    
    # (모델 1: SDXL 로드)
    print("Loading Stable Diffusion XL pipeline...")
    models["image_pipe"] = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    print("SDXL 로드 완료.")

    # (모델 2: GPT-4o API 사용)
    print("KoAlpaca 대신 GPT-4o API를 사용합니다.")
    
    # (모델 3: Real-ESRGAN 로드 - 파일 경로 수정)
    print("Loading Real-ESRGAN model...")
    try:
        # Dockerfile에서 지정한 독립 경로 사용
        model_path = "/app/esrgan/RealESRGAN_x4plus.pth"
        
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        models["upsampler"] = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=esrgan_model,
            tile=500,
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

# (✨ v38 수정) DB에서 강아지 정보를 가져오는 함수 (실제 로직)
# 🚨 Task 5 반영: homeprotectionsub02 테이블에서 이미지 파일명을 조회하도록 수정
async def get_dog_details(dog_uid: int) -> Dog:
    db = await get_db_connection()
    
    # 1. homeprotection (주요 정보 및 s_pic01) 조회
    main_query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog_data = await db.fetch_one(main_query)
    
    if not dog_data:
        raise HTTPException(status_code=404, detail=f"UID {dog_uid}에 해당하는 강아지 정보를 DB에서 찾을 수 없습니다.")

    # 2. homeprotectionsub02에서 파일명 조회
    # puid == dog_uid를 조건으로, num으로 정렬하여 갤러리 파일 목록을 가져옵니다.
    image_query = sub02_table.select().where(sub02_table.c.puid == dog_uid).order_by(sub02_table.c.num)
    image_data_list = await db.fetch_all(image_query)
    
    # 3. 파일명 리스트 추출
    image_filenames = [row['s_pic01'] for row in image_data_list]

    # 4. Dog Pydantic 모델 생성 시 파일명 리스트 추가
    return Dog(**dog_data, image_filenames=image_filenames)


# --- 헬퍼 함수 (Image/Text) ---

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# (GPT-4o API를 사용하는 텍스트 생성 - v35 최종 버전, Task C 반영)
def generate_dog_text(dog: Dog) -> str:
    def clean_text(text):
        if not text: return ""
        # HTML 태그 제거 로직은 그대로 유지
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    # DB에서 가져온 강아지 정보
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

    # 🚨 Task C 반영: 수정된 시스템 프롬프트 (간결한 항목 스타일 강제)
    system_prompt = """
    당신은 유기견의 입양 공고를 간결하게 작성하는 AI 전문가입니다.
    요청된 정보와 특징을 바탕으로, 감성적인 설명이나 장황한 문장 대신 **핵심 정보만 포함**하는 공고 스타일의 프로필 텍스트를 생성하세요.

    **[생성 규칙]**
    1. 텍스트는 **5줄 이내**로 작성되어야 합니다.
    2. 출력은 아래 요청 항목과 같이 **항목별 단문** 형태로 구성되어야 합니다.
    3. 입양 문의 방법은 마지막 줄에 **반드시** '인스타그램 @lovely4puppies에서 확인하세요.'와 같은 형태로 포함합니다.
    4. 정보에 없는 내용은 절대 지어내지 않습니다.
    """

    # 🚨 Task C 반영: 수정된 사용자 요청 (항목별 출력을 유도)
    user_content = f"""
    [강아지 정보]:
    {dog_info}

    위 정보를 기반으로 다음 4가지 항목을 포함하는 간결한 공고문을 작성해 주세요.
    1. 이름/성별/몸무게 (예: '해리 / 여 / 10kg')
    2. 특징 및 성격 (예: '순둥하고 애교 많음')
    3. 특이사항 및 건강 상태 (예: '중성화 완료, 화재 경험 극복')
    4. 입양 문의 (출력 규칙 3번 반영)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    print("GPT-4o 텍스트 생성 시작...")
    
    try:
        # 🚨 GPT 오류 해결: openai>=1.0.0 버전 클라이언트 사용
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not client.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 비용 및 속도 개선을 위해 mini 모델 사용
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        generated_text = response.choices[0].message.content.strip()
        print("GPT-4o 텍스트 생성 완료.")
            
    except Exception as e:
        print(f"🚨 GPT-4o API 호출 중 오류 발생: {type(e).__name__}: {e}")
        generated_text = "GPT-4o API 오류로 소개글을 생성할 수 없습니다."
            
    return generated_text

# (이미지 선별 로직 - 5개 파일 중 최적 이미지 선정)
# 🚨 Task D 반영 및 개선: 마스크 크기(70%)와 선명도(30%) 가중치 조합 로직 적용
# 🚨 Task 5 반영: Dog 모델의 image_filenames 속성을 사용하여 파일 목록 가져옴
async def select_best_image(dog: Dog) -> Tuple[Union[Image.Image, None], Union[str, None]]:
    best_input_image_pil = None
    best_score = -1 
    original_rgb_image_base64 = None

    # 🚨 Task 5 반영: s_pic01과 sub02에서 가져온 목록을 병합하여 사용
    image_filenames = []
    
    # 1. 대표 사진 s_pic01을 목록에 추가 (최우선)
    if dog.s_pic01:
        image_filenames.append(dog.s_pic01)
        
    # 2. sub02에서 가져온 갤러리 이미지 파일 목록을 추가
    image_filenames.extend(dog.image_filenames) # Dog 모델의 image_filenames 속성 사용

    if not image_filenames:
        print(f"[{dog.uid}] !! 유효한 이미지 파일이 없습니다.")
        return None, None
    
    remover_session = models.get("remover")
    if not remover_session:
        raise RuntimeError("🚨 rembg 세션이 로드되지 않았습니다.")

    print(f"[{dog.uid}] 최적 이미지 선별 시작...")
    
    # ⭐️ Task D 개선을 위한 정규화 기준값 (실제 환경에 따라 조정 가능)
    # 이미지 크기 중요도를 높이기 위해 마스크 크기 기준값 MAX_MASK_SIZE를 사용
    MAX_MASK_SIZE = 100000 
    MAX_FOCUS_SCORE = 1000 

    for filename in image_filenames:
        if not filename or filename.strip() == "":
            continue
        try:
            image_url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{filename}"
            
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()
            
            input_image_pil = Image.open(response.raw).convert("RGB")
            
            # 1. 선명도(Focus) 측정 (Laplacian Variance)
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
            
            # 3. 종합 점수 계산 (마스크 크기(70%) + 선명도(30%) 가중치)
            
            # 정규화: 기준값으로 나누어 0~1 사이의 값으로 만듦 (기준값 초과 시 1로 간주)
            normalized_mask = min(mask_size, MAX_MASK_SIZE) / MAX_MASK_SIZE
            normalized_focus = min(focus_measure, MAX_FOCUS_SCORE) / MAX_FOCUS_SCORE
            
            # ⭐️ 개선된 로직: 크기 70%, 선명도 30% 가중치 적용
            composite_score = (normalized_mask * 0.7) + (normalized_focus * 0.3)
            
            if composite_score > best_score:
                print(f"     >>> ★★★ 새 최적 이미지 발견! (점수: {composite_score:.4f}, 마스크: {mask_size}, 선명도: {focus_measure:.2f}, 파일: {filename})")
                best_score = composite_score 
                best_input_image_pil = input_image_pil
                
                # 원본 이미지 Base64 저장 로직 (최적 이미지가 바뀔 때마다 업데이트)
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

    if best_input_image_pil:
        try:
            # 2. Real-ESRGAN으로 화질 복원 
            cv2_image = pil_to_cv2(best_input_image_pil)
            upscaled_image_cv2, _ = models["upsampler"].enhance(cv2_image, outscale=4)
            upscaled_image_pil = cv2_to_pil(upscaled_image_cv2)
            print("화질 복원 완료.")

            # 3. 복원된 이미지의 배경 제거
            print("배경 제거(rembg) 시작...")
            remover_session = models.get("remover") 
            removed_bg_image = remove(
                upscaled_image_pil,
                session=remover_session,
                alpha_matting=True
            )
            print("배경 제거 완료.")

            # 4. 텍스트 생성 (GPT-4o)
            generated_text = generate_dog_text(dog)
            
            # 5. Pillow 템플릿 합성 (간소화)
            print("Pillow 템플릿 합성 시작...")
            template_width = 800
            template_height = 1200
            template = Image.new('RGB', (template_width, template_height), (255, 255, 255))
            draw = ImageDraw.Draw(template)

            # 폰트 로딩 (NanumGothicBold.ttf가 /app/ 경로에 있다는 가정)
            try:
                font_title = ImageFont.truetype("/app/NanumGothic-Bold.ttf", 40)
                font_body = ImageFont.truetype("/app/NanumGothic-Regular.ttf", 24)
            except IOError:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
                print("!! 폰트 파일 로드 실패. 기본 폰트 사용.")


            # 이미지 배치 및 텍스트 로직은 이전 코드를 유지
            img_height = int(template_width * (removed_bg_image.height / removed_bg_image.width))
            image_to_template = removed_bg_image.resize((template_width, img_height))
            template.paste(image_to_template, (0, 0), image_to_template) 

            text_y_position = img_height + 30
            draw.text((30, text_y_position), dog.subject, font=font_title, fill=(0,0,0))
            text_y_position += 60

            lines = textwrap.wrap(generated_text, width=60)
            for line in lines:
                draw.text((30, text_y_position), line, font=font_body, fill=(50, 50, 50))
                text_y_position += 30

            buffered = io.BytesIO()
            template.save(buffered, format="PNG")
            final_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            print("Pillow 템플릿 합성 완료.")

        except Exception as e:
            print(f"[{dog.uid}] !! 이미지/텍스트 처리 중 오류: {e}. 원본 이미지를 반환합니다.")
            final_image_base64 = original_rgb_image_base64 or "Error: Template composition failed."
            generated_text = generated_text or "프로필 생성 중 오류가 발생했습니다."
            
    else:
        print(f"[{dog.uid}] !! 치명적 오류: 유효한 이미지가 없어 프로필 생성을 중단합니다.")
        generated_text = "프로필을 생성할 수 없습니다: 유효한 원본 이미지가 없습니다."
        final_image_base64 = ""
        
    return {
        "profile_text": generated_text,
        "profile_image_base64": final_image_base64
    }

@app.get("/api/dogs/{dog_uid}", response_model=Dog)
async def get_dog_details_api(dog_uid: int):
    return await get_dog_details(dog_uid)
