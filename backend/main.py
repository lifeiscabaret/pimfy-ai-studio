import torchvision
try:
    # torchvision 0.17+ 버전에서 삭제된 functional_tensor를 functional로 우회 연결
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = F
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

# --- [필수] PyTorch 모델 로딩 보안 패치 ---
_original_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load
# ---------------------------------------

import databases
import sqlalchemy
from rembg import new_session, remove
from realesrgan import RealESRGANer 
from basicsr.archs.rrdbnet_arch import RRDBNet 
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import openai 
import httpx 

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/inday_fileinfo/img")
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "https://www.pimfyvirus.com")

database = databases.Database(DATABASE_URL) if DATABASE_URL else None
metadata = sqlalchemy.MetaData()

# --- DB 테이블 정의 ---
dogs_table = sqlalchemy.Table(
    "homeprotection", metadata,
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
    sqlalchemy.Column("addinfo11", sqlalchemy.String(250)),
    sqlalchemy.Column("addinfo19", sqlalchemy.String(250)),
)

sub02_table = sqlalchemy.Table(
    "homeprotectionsub02", metadata,
    sqlalchemy.Column("puid", sqlalchemy.Integer), 
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),
    sqlalchemy.Column("num", sqlalchemy.Integer), 
)

# --- 데이터 모델 ---
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

# --- 앱 초기화 ---
models = {}
app = FastAPI()

# ⭐️ GPU 모드 확인 및 설정
if torch.cuda.is_available():
    device = "cuda"
    gpu_id = 0
    print(f"🚀 [System] GPU 모드 활성화: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    gpu_id = None
    print("⚠️ [System] 경고: GPU를 찾을 수 없습니다. CPU로 실행됩니다.")

SDXL_SERVICE_URL = "http://sdxl-service:8001/generate/background"

@app.on_event("startup")
def load_models_and_db():
    print("🚀 AI 서버 시작: 모델 로딩 중...")
    
    # (1) Real-ESRGAN 로드
    print("Loading Real-ESRGAN PyTorch Model...")
    try:
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = "/app/esrgan/RealESRGAN_x4plus.pth"

        models["upsampler"] = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model_arch,
            tile=0,       #️ V100 풀파워
            tile_pad=10,
            pre_pad=0,
            half=True if device == "cuda" else False, 
            gpu_id=gpu_id
        )
        print("✅ Real-ESRGAN PyTorch Model loaded successfully.")
    except Exception as e:
        print(f"🚨 Real-ESRGAN Load Failed: {e}")

    # (2) rembg 로드 (BiRefNet 적용)
    print("Loading rembg session (BiRefNet)...")
    try:
        # 털 묘사 업그레이드 모델 (최초 실행 시 다운로드 시간 소요)
        models["remover"] = new_session(model_name="birefnet-general")
        print("✅ rembg 세션 로드 완료 (Model: birefnet-general).")
    except Exception as e:
        print(f"🚨 BiRefNet 로드 실패: {e}. 기존 isnet으로 폴백합니다.")
        models["remover"] = new_session(model_name="isnet-general-use")

    print("--- 모든 AI 모델 로딩 완료 ---")

@app.on_event("shutdown")
async def shutdown_db_client():
    if database and database.is_connected:
        await database.disconnect()

async def get_db_connection():
    if database and not database.is_connected:
        await database.connect()
    return database

# --- Helper Functions ---
async def get_dog_details(dog_uid: int) -> Dog:
    db = await get_db_connection()
    if not db: raise HTTPException(status_code=500, detail="DB Fail")
    main_query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog_data = await db.fetch_one(main_query)
    if not dog_data: raise HTTPException(status_code=404, detail="Dog Not Found")
    image_query = sub02_table.select().where(sub02_table.c.puid == dog_uid).order_by(sub02_table.c.num)
    image_data_list = await db.fetch_all(image_query)
    image_filenames = [row['s_pic01'] for row in image_data_list]
    return Dog(**dog_data, image_filenames=image_filenames)

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# 텍스트 테두리(Stroke) 그리기 함수 
def draw_text_with_stroke(draw, x, y, text, font, fill_color, stroke_color, stroke_width):
    for dx, dy in [(sx, sy) for sx in range(-stroke_width, stroke_width + 1) for sy in range(-stroke_width, stroke_width + 1) if sx * sx + sy * sy <= stroke_width * stroke_width]:
        draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=fill_color)

def get_text_width(draw, text, font):
    max_width = 0
    if not text: return 0
    for line in text.split('\n'):
        try:
            width = draw.textlength(line, font=font)
        except:
            width = len(line) * (font.size * 0.6)
        if width > max_width: max_width = width
    return max_width

def remove_emojis(text):
    if not text: return ""
    return re.sub(r'[^\w\s,.\-?!@#%&()가-힣/]', '', text).strip()

def extract_contact_info(text):
    if not text: return "문의: 자세한 내용은 공고 원문 참조"
    insta_id_match = re.search(r'@[a-zA-Z0-9_.]+', text)
    if insta_id_match: return f"인스타 {insta_id_match.group()}"
    insta_url_match = re.search(r'instagram\.com/([a-zA-Z0-9_.]+)', text)
    if insta_url_match: return f"인스타 @{insta_url_match.group(1)}"
    url_match = re.search(r'(https?://[^\s]+)', text)
    if url_match:
        if "instagram" in url_match.group(0): return "인스타 링크 참조"
        return "SNS 링크 참조"
    phone_match = re.search(r'010-?[\d]{3,4}-?[\d]{4}', text)
    if phone_match: return f"문의 Tel: {phone_match.group()}"
    return "문의: 자세한 내용은 공고 원문 참조"

#  (Feathering) 함수
def apply_feathering(pil_img, blur_radius=2):
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    r, g, b, a = pil_img.split()
    # 알파 채널에 블러처리 -> 경계 흐릿하게.
    a_blurred = a.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.merge("RGBA", (r, g, b, a_blurred))

async def call_sdxl_service(base64_dog_image: str, dog_info: dict) -> Image.Image:
    color_hint = "pastel pink" 
    prompt_detail = f"Minimalist studio background suitable for {dog_info.get('name', 'a dog')}."
    payload = {"base64_dog_image": base64_dog_image, "prompt": prompt_detail, "color_hint": color_hint}
    print(f"Calling SDXL service... Hint: {color_hint}")
    
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            response = await client.post(SDXL_SERVICE_URL, json=payload)
            response.raise_for_status()  
            result = response.json()
            base64_bg = result.get("base64_background_image")
            if not base64_bg: raise ValueError("No bg image")
            return Image.open(io.BytesIO(base64.b64decode(base64_bg))).convert("RGB")
        except Exception as e:
            print(f"🚨 SDXL Error: {e}")
            return Image.new('RGB', (800, 1200), (255, 255, 255))

def generate_dog_text(dog: Dog) -> List[str]: 
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        return remove_emojis(text)

    raw_subject = dog.subject if dog.subject else ""
    if '/' in raw_subject: raw_name = raw_subject.split('/')[0] 
    else: raw_name = raw_subject 
    dog_name_only = clean_text(raw_name).strip()
    
    display_age = dog.addinfo05 if dog.addinfo05 and not dog.addinfo05.isdigit() else f"{dog.addinfo05[:4]}년 {dog.addinfo05[4:]}월생" if dog.addinfo05 and len(dog.addinfo05)==6 else "정보 없음"
    
    basic_info = [
        f"이름: {dog_name_only}",
        f"성별: {clean_text(dog.addinfo03)}",
        f"출생시기: {display_age}", 
        f"몸무게: {clean_text(dog.addinfo07)}kg",
        f"중성화: {clean_text(dog.addinfo04)}",
    ]
    
    story_data = f"이름:{dog_name_only}, 성격:{clean_text(dog.addinfo10)}({clean_text(dog.addinfo08)}), 사연:{clean_text(dog.addinfo09)}"
    messages = [{"role": "system", "content": "유기견 입양 홍보 문구 2줄 작성. 감성적, 간결하게. 이모티콘 사용 금지."}, {"role": "user", "content": f"정보: {story_data}"}]
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=500)
        generated_story = remove_emojis(res.choices[0].message.content.strip()) 
    except:
        generated_story = "따뜻한 가족을 기다립니다."

    contact_source = ""
    if dog.addinfo11: contact_source += dog.addinfo11 
    if dog.addinfo15: contact_source += " " + dog.addinfo15
    if dog.addinfo12: contact_source += " " + dog.addinfo12
    final_contact_info = extract_contact_info(contact_source)

    return basic_info + [generated_story, dog_name_only, final_contact_info] 

#  사진 선별 로직 ( 중앙 집중)
async def select_best_image(dog: Dog) -> Tuple[Union[Image.Image, None], Union[str, None]]:
    best_img, best_score, best_b64 = None, -999, None
    imgs = [dog.s_pic01] + dog.image_filenames if dog.s_pic01 else dog.image_filenames
    imgs = list(dict.fromkeys([x for x in imgs if x and x.strip()])) # 중복 제거
    if not imgs: return None, None
    
    remover_session = models.get("remover")
    print(f"[{dog.uid}] 이미지 정밀 선별 중 ({len(imgs)}장)...")
    
    for fname in imgs:
        try:
            url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{fname}"
            res = requests.get(url, stream=True, timeout=5)
            res.raise_for_status()
            img = Image.open(res.raw).convert("RGB")
            w, h = img.size

            # 속도를 위해 리사이징 & Alpha Matting OFF
            img_small = img.resize((300, int(300*h/w)))
            no_bg_small = remove(img_small, session=remover_session, alpha_matting=False)
            
            alpha = np.array(no_bg_small.split()[3])
            if cv2.countNonZero(alpha) == 0: continue

            coords = cv2.findNonZero(alpha)
            x, y, box_w, box_h = cv2.boundingRect(coords)
            
            mask_area = box_w * box_h
            total_area = img_small.width * img_small.height
            mask_ratio = mask_area / total_area
            
            score = 0
            
            # 1. 크기 점수 (꽉 찬 사진 우대, 10% 미만 탈락)
            if mask_ratio < 0.10: score = -10.0
            else: score += min(mask_ratio * 5.0, 5.0) # 클수록 점수 (최대 5점)

            # 2. 중앙 집중도 점수
            center_x = x + box_w / 2
            center_y = y + box_h / 2
            img_center_x = img_small.width / 2
            img_center_y = img_small.height / 2
            dist_norm = ((center_x - img_center_x)**2 + (center_y - img_center_y)**2)**0.5
            max_dist = (img_small.width**2 + img_small.height**2)**0.5
            score += (1 - (dist_norm / max_dist)) * 3.0

            # 3. 세로 사진 우대
            if h > w: score += 2.0

            # 4. 하단 잘림 체크 (다리/발 잘린 사진 감점)
            if (y + box_h) > (img_small.height * 0.98): score -= 2.0 

            if score > best_score:
                best_score = score
                best_img = img
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                best_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except: continue
    return best_img, best_b64

# --- 메인 API ---
@app.post("/api/v1/generate-real-profile", response_model=dict)
async def generate_real_profile(request: RealProfileRequest):
    if "upsampler" not in models: raise HTTPException(status_code=503, detail="Model Loading")
    dog = await get_dog_details(request.dog_uid)
    best_img, orig_b64 = await select_best_image(dog)
    if not best_img: return {"profile_text": "이미지 없음", "profile_image_base64": ""}

    try:
        # 1. Upscaling
        cv2_img = pil_to_cv2(best_img)
        output, _ = models["upsampler"].enhance(cv2_img, outscale=4)
        upscaled_pil = cv2_to_pil(output)
        print("✅ 화질 복원 완료")

        # 2. Background Removal (BiRefNet + Feathering)
        # ⭐️ alpha_matting=False (속도) / Feathering (부드러움)
        no_bg = remove(upscaled_pil, session=models["remover"], alpha_matting=False)
        no_bg = apply_feathering(no_bg, blur_radius=2)
        print("✅ 배경 제거 및 페더링 완료")
        
        # 3. SDXL Background
        buf = io.BytesIO()
        no_bg.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        bg_img = await call_sdxl_service(b64_png, {"name": dog.subject})
        
        # 4. Template Generation
        text_result = generate_dog_text(dog)
        texts = text_result[0:5] 
        story = text_result[5]
        dog_name_only = text_result[6] 
        contact_info = text_result[7]

        template_w, template_h = 1080, 1350
        template = bg_img.resize((template_w, template_h))
        draw = ImageDraw.Draw(template)
        
        try:
            ft = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 80)
            fb = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 38)
            fc = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 30) 
        except: ft = fb = fc = ImageFont.load_default()

        t_txt = f"{dog_name_only}의 가족을 찾습니다."
        tw = get_text_width(draw, t_txt, ft)
        draw_text_with_stroke(draw, (template_w-tw)/2, 60, t_txt, ft, (255,255,255), (0,0,0), 3)
        
        orig_w, orig_h = no_bg.size
        disp_w = template_w
        disp_h = int(orig_h * (disp_w / orig_w))
        
        MAX_IMG_H = 600
        if disp_h > MAX_IMG_H:
            disp_h = MAX_IMG_H
            disp_w = int(orig_w * (disp_h / orig_h))
        
        paste_img = no_bg.resize((disp_w, disp_h))
        template.paste(paste_img, ((template_w-disp_w)//2, 180), paste_img)
        
        cy = 180 + disp_h + 60
        for i, line in enumerate(texts): 
            w = get_text_width(draw, line, fb)
            draw.text(((template_w-w)/2, cy), line, font=fb, fill=(50,50,50))
            cy += 50 
        
        cy += 30
        for line in textwrap.wrap(story, width=40):
            w = get_text_width(draw, line, fb)
            draw.text(((template_w-w)/2, cy), line, font=fb, fill=(0,0,0))
            cy += 50
            
        # ️ SNS/연락처 정보 출력 (테두리 추가)
        cw = get_text_width(draw, contact_info, fc)
        draw_text_with_stroke(
            draw, 
            (template_w-cw)/2, 
            template_h - 80, 
            contact_info, 
            fc, 
            (100, 100, 100), # 내부 글씨 색 (진한 회색)
            (255, 255, 255), # 테두리 색 (흰색)
            2
        )

        buf = io.BytesIO()
        template.save(buf, format="PNG")
        final_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"profile_text": '\n'.join(texts), "profile_image_base64": final_b64}
        
    except Exception as e:
        print(f"🚨 Processing Error: {e}")
        return {"profile_text": "Error", "profile_image_base64": orig_b64}
