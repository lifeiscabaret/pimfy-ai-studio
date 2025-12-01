import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = F
# ---------------------------------------

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
# 👇 CORS 미들웨어 임포트
from fastapi.middleware.cors import CORSMiddleware 
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
from typing import Optional, List, Tuple, Union, Dict
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageColor

# ⭐️ HEIC 포맷 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ HEIC Image Support Enabled.")
except ImportError:
    print("⚠️ pillow-heif not found. HEIC images might cause errors.")

import databases
import sqlalchemy
from rembg import new_session, remove 
from realesrgan import RealESRGANer 
from basicsr.archs.rrdbnet_arch import RRDBNet 
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
    addinfo01: Optional[str] = None 
    addinfo02: Optional[str] = None 
    addinfo03: Optional[str] = None
    addinfo04: Optional[str] = None
    addinfo05: Optional[str] = None
    addinfo07: Optional[str] = None
    addinfo08: Optional[str] = None
    addinfo09: Optional[str] = None
    addinfo10: Optional[str] = None
    addinfo11: Optional[str] = None
    addinfo12: Optional[str] = None
    addinfo15: Optional[str] = None
    addinfo19: Optional[str] = None

class RealProfileRequest(BaseModel):
    dog_uid: int

# --- 앱 초기화 ---
models = {}
app = FastAPI()

# ⭐️ [필수] CORS 설정 추가 (여기!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 주소 허용 (로컬 테스트용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    # 1. Real-ESRGAN
    print("Loading Real-ESRGAN PyTorch Model...")
    try:
        model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = "/app/esrgan/RealESRGAN_x4plus.pth"
        models["upsampler"] = RealESRGANer(
            scale=4, model_path=model_path, model=model_arch, tile=0, tile_pad=10, pre_pad=0,
            half=True if device == "cuda" else False, gpu_id=gpu_id
        )
        print("✅ Real-ESRGAN Loaded.")
    except Exception as e:
        print(f"🚨 Real-ESRGAN Failed: {e}")

    # 2. Rembg
    print("Loading Rembg...")
    try:
        models["remover"] = new_session(model_name="isnet-general-use")
        print("✅ Rembg Loaded.")
    except:
        models["remover"] = new_session()

    print("--- 모든 AI 모델 로딩 완료 ---")

@app.on_event("shutdown")
async def shutdown_db_client():
    if database and database.is_connected: await database.disconnect()

async def get_db_connection():
    if database and not database.is_connected: await database.connect()
    return database

# --- Helper Functions ---
def resize_image_if_too_large(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

# ⭐️ [핵심 수정] "정사각형 캔버스"에 이미지를 "가운데 정렬"하는 함수
# 이렇게 하면 겉모양은 무조건 정사각형(깔끔함)이 되고, 사진은 안 잘립니다.
def create_framed_image(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    
    # 1. 정사각형 캔버스 크기 결정 (사진의 가장 긴 변 기준)
    # 조금 넉넉하게 잡아서 해상도 유지
    canvas_size = max(w, h)
    
    # 2. 흰색 정사각형 캔버스 생성
    square_canvas = Image.new('RGB', (canvas_size, canvas_size), 'white')
    
    # 3. 중앙 좌표 계산
    offset_x = (canvas_size - w) // 2
    offset_y = (canvas_size - h) // 2
    
    # 4. 사진 붙이기 (잘림 없음!)
    square_canvas.paste(pil_img, (offset_x, offset_y))
    
    # 5. 테두리 추가 (캔버스 크기의 3% 정도만) - 폴라로이드 느낌 살짝
    border_size = int(canvas_size * 0.03)
    framed_img = ImageOps.expand(square_canvas, border=border_size, fill='white')
    
    return framed_img

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

def draw_text_with_stroke(draw, x, y, text, font, fill_color, stroke_color, stroke_width):
    for dx, dy in [(sx, sy) for sx in range(-stroke_width, stroke_width + 1) for sy in range(-stroke_width, stroke_width + 1) if sx * sx + sy * sy <= stroke_width * stroke_width]:
        draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=fill_color)

def get_text_width(draw, text, font):
    try: return draw.textlength(text, font=font)
    except: return len(text) * (font.size * 0.6)

def remove_emojis(text):
    if not text: return ""
    return re.sub(r'[^\w\s,.\-?!@#%&()가-힣/]', '', text).strip()

async def call_sdxl_service(base64_dog_image: str, dog_info: dict) -> Image.Image:
    color_hint = "warm cream and white"
    prompt_detail = "A minimalist aesthetic background, warm sunlight shadows on a white wall, clean interior, cozy atmosphere, high quality, soft focus, instagram vibe."

    payload = {"base64_dog_image": base64_dog_image, "prompt": prompt_detail, "color_hint": color_hint}
    
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            response = await client.post(SDXL_SERVICE_URL, json=payload)
            response.raise_for_status()  
            result = response.json()
            base64_bg = result.get("base64_background_image")
            if not base64_bg: raise ValueError("No bg image")
            return Image.open(io.BytesIO(base64.b64decode(base64_bg))).convert("RGB")
        except Exception as e:
            return Image.new('RGB', (1080, 1350), (250, 245, 240)) 

def generate_dog_text(dog: Dog) -> Dict: 
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        return remove_emojis(text)

    raw_name = dog.subject.split('/')[0] if '/' in dog.subject else dog.subject
    dog_name_only = clean_text(raw_name).strip()
    display_age = dog.addinfo05 if dog.addinfo05 and not dog.addinfo05.isdigit() else "정보 없음"
    
    basic_info_lines = [
        f"이름: {dog_name_only}",
        f"성별: {clean_text(dog.addinfo03)}",
        f"출생시기: {display_age}", 
        f"몸무게: {clean_text(dog.addinfo07)}kg",
        f"중성화: {clean_text(dog.addinfo04)}",
    ]
    
    info_source = [dog.addinfo08, dog.addinfo09, dog.addinfo10, dog.addinfo01]
    story_data = f"이름:{dog_name_only}, " + " ".join([clean_text(x) for x in info_source if x])
    
    system_prompt = """
    당신은 입양을 기다리는 유기견입니다. 미래의 가족에게 보내는 짧은 편지를 작성하세요.
    규칙: 1인칭('저', '제'), 다정한 존댓말(해요체). 2~3문장.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"강아지 정보: {story_data}"}
    ]
    
    generated_story = ""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=300)
        generated_story = remove_emojis(res.choices[0].message.content.strip())
    except Exception as e:
        generated_story = f"안녕하세요, 저는 {dog_name_only}예요! 저의 평생 가족이 되어주실 분을 기다리고 있어요."

    return {
        "basic_info": basic_info_lines,
        "story": generated_story,
        "name": dog_name_only
    }

async def select_best_image(dog: Dog) -> Union[Image.Image, None]:
    best_img, best_score = None, -9999
    imgs = list(dict.fromkeys([x for x in ([dog.s_pic01] + dog.image_filenames) if x and x.strip()]))
    if not imgs: return None
    
    remover = models.get("remover")
    if not remover: return None 

    print(f"[{dog.uid}] AI 스마트 선별 중 ({len(imgs)}장)...")
    
    for fname in imgs:
        try:
            url = f"{SITE_BASE_URL}{IMAGE_BASE_PATH}/{fname}"
            res = requests.get(url, stream=True, timeout=5)
            res.raise_for_status()
            img = Image.open(res.raw).convert("RGB")
            img = ImageOps.exif_transpose(img)
            
            w, h = img.size
            if w < 250 or h < 250: continue 

            small_w = 320
            small_h = int(h * (small_w / w))
            img_small = img.resize((small_w, small_h))
            
            no_bg = remove(img_small, session=remover, alpha_matting=False)
            
            alpha = np.array(no_bg.split()[3])
            if cv2.countNonZero(alpha) == 0: continue 
            
            coords = cv2.findNonZero(alpha)
            x, y, box_w, box_h = cv2.boundingRect(coords)
            
            mask_area = box_w * box_h
            total_area = small_w * small_h
            score_size = (mask_area / total_area) * 100 

            center_x = x + box_w / 2
            center_y = y + box_h / 2
            dist_from_center = ((center_x - small_w/2)**2 + (center_y - small_h/2)**2)**0.5
            max_dist = (small_w**2 + small_h**2)**0.5
            score_center = (1 - (dist_from_center / max_dist)) * 50 
            
            img_gray = cv2.cvtColor(np.array(img_small), cv2.COLOR_RGB2GRAY)
            masked_gray = img_gray[y:y+box_h, x:x+box_w]
            if masked_gray.size > 0:
                laplacian_var = cv2.Laplacian(masked_gray, cv2.CV_64F).var()
                score_sharp = min(laplacian_var, 500) / 10 
            else:
                score_sharp = 0
                
            score_penalty = 0
            if h > w * 2.2: score_penalty = 30 
            
            total_score = score_size + score_center + score_sharp - score_penalty
            if h > w: total_score += 20
            
            if total_score > best_score:
                best_score = total_score
                best_img = img
                
        except Exception as e:
            continue
            
    return best_img

# =========================================================
# 1. 자동 프로필 생성 (핌피바이러스 공고)
# =========================================================
@app.post("/api/v1/generate-real-profile", response_model=dict)
async def generate_real_profile(request: RealProfileRequest):
    if "upsampler" not in models: raise HTTPException(status_code=503, detail="Model Loading")
    try:
        dog = await get_dog_details(request.dog_uid)
        best_img = await select_best_image(dog)
        if not best_img: return {"profile_text": "이미지 없음", "profile_image_base64": ""}
        
        best_img = resize_image_if_too_large(best_img)

        cv2_img = pil_to_cv2(best_img)
        output, _ = models["upsampler"].enhance(cv2_img, outscale=4)
        upscaled_pil = cv2_to_pil(output)

        # ⭐️ [적용] 정사각형 액자에 이미지 중앙 배치 (Fit)
        processed_img = create_framed_image(upscaled_pil)
        
        buf = io.BytesIO()
        processed_img.save(buf, format="PNG") 
        b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        bg_img = await call_sdxl_service(b64_png, {"name": dog.subject})
        
        text_data = generate_dog_text(dog)
        
        template_w, template_h = 1080, 1350
        template = bg_img.resize((template_w, template_h))
        draw = ImageDraw.Draw(template)
        
        try:
            ft = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 80)
            fb = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 38)
        except: 
            ft = fb = ImageFont.load_default()

        t_txt = f"{text_data['name']}의 가족을 찾습니다."
        tw = get_text_width(draw, t_txt, ft)
        draw_text_with_stroke(draw, (template_w-tw)/2, 60, t_txt, ft, (255,255,255), (0,0,0), 3)
        header_height = 180

        lines = text_data['basic_info'] + textwrap.wrap(text_data['story'], width=35)
        line_height = 50
        text_total_height = (len(lines) * line_height) + 50 
        footer_margin = 100 
        
        available_h = template_h - header_height - text_total_height - footer_margin
        
        p_w, p_h = processed_img.size
        # 가로 900으로 고정 (정사각형이므로 세로도 900이 됨)
        target_w = 900
        target_h = int(p_h * (target_w / p_w))
        
        if target_h > available_h:
            target_h = available_h
            target_w = int(p_w * (target_h / p_h))

        paste_img = processed_img.resize((target_w, target_h))
        template.paste(paste_img, ((template_w-target_w)//2, header_height))
        
        cy = header_height + target_h + 40
        for line in lines:
            w = get_text_width(draw, line, fb)
            draw_text_with_stroke(draw, (template_w-w)/2, cy, line, fb, (50,50,50), (255,255,255), 2)
            cy += line_height
        
        buf = io.BytesIO()
        template = template.convert("RGB")
        template.save(buf, format="JPEG", quality=90, optimize=True)
        final_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        torch.cuda.empty_cache()
        
        return {
            "profile_text": '\n'.join(text_data['basic_info'] + [text_data['story']]), 
            "profile_image_base64": final_b64
        }
        
    except Exception as e:
        print(f"🚨 Auto Profile Error: {e}")
        return {"profile_text": "Error", "profile_image_base64": ""}

# =========================================================
# 2. 입양/임보 프로필 (수동 입력)
# =========================================================
@app.post("/api/v1/generate-adoption-profile", response_model=dict)
async def generate_adoption_profile(
    image: UploadFile = File(...),
    name: str = Form(...),
    age: str = Form(...),
    personality: str = Form(...),
    features: str = Form(...)
):
    if "upsampler" not in models: raise HTTPException(status_code=503, detail="Model Loading")

    try:
        contents = await image.read()
        if len(contents) == 0: raise ValueError("Uploaded file is empty.")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = ImageOps.exif_transpose(img) 
        img = resize_image_if_too_large(img)

        cv2_img = pil_to_cv2(img)
        output, _ = models["upsampler"].enhance(cv2_img, outscale=4)
        upscaled_pil = cv2_to_pil(output)

        # ⭐️ [적용] 정사각형 액자에 이미지 중앙 배치 (Fit)
        processed_img = create_framed_image(upscaled_pil)
        
        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")
        bg_img = await call_sdxl_service(b64_png, {"name": name})

        # 프롬프트: 성격/특징을 녹여내도록 요청
        story_data = f"이름:{name}, 나이:{age}, 성격:{personality}, 특징:{features}"
        system_prompt = """
        당신은 입양을 기다리는 강아지입니다. 미래의 가족에게 보내는 편지를 작성하세요.
        규칙:
        1. 제공된 성격과 특징을 **모두 자연스럽게 녹여서** 하나의 이야기로 만드세요.
        2. 시점은 '저', '제'를 사용한 1인칭 시점입니다.
        3. 말투는 사랑스럽고 다정한 '해요체' 존댓말을 사용하세요.
        4. 길이는 2~3문장으로 제한합니다.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"내 정보: {story_data}"}
        ]
        
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=300)
            generated_story = remove_emojis(res.choices[0].message.content.strip())
        except:
            generated_story = f"안녕하세요! 저는 {name}이에요. 사랑 넘치는 가족을 기다리고 있어요!"

        template_w, template_h = 1080, 1350
        template = bg_img.resize((template_w, template_h))
        draw = ImageDraw.Draw(template)
        
        try:
            ft = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 80)
            fb = ImageFont.truetype("/app/KyoboHandwriting2021sjy.otf", 38)
        except: ft = fb = ImageFont.load_default()

        t_txt = f"{name}의 가족을 찾습니다."
        tw = get_text_width(draw, t_txt, ft)
        draw_text_with_stroke(draw, (template_w-tw)/2, 60, t_txt, ft, (255,255,255), (0,0,0), 3)
        
        # ⭐️ [핵심 수정] 성격/특징 날것 출력 삭제! 이름/나이만 출력
        info_lines = [f"이름: {name}", f"나이: {age}"]
        
        # AI 스토리만 붙임
        lines = info_lines + textwrap.wrap(generated_story, width=35)
        
        header_height = 180
        line_height = 50
        text_total_height = (len(lines) * line_height) + 50
        footer_margin = 100
        
        available_h = template_h - header_height - text_total_height - footer_margin
        p_w, p_h = processed_img.size
        
        # 가로 900 고정 (정사각형이므로)
        target_w = 900
        target_h = int(p_h * (target_w / p_w))
        
        if target_h > available_h:
            target_h = available_h
            target_w = int(p_w * (target_h / p_h))

        paste_img = processed_img.resize((target_w, target_h))
        template.paste(paste_img, ((template_w-target_w)//2, header_height))
        
        cy = header_height + target_h + 40
        for line in lines:
            w = get_text_width(draw, line, fb)
            draw_text_with_stroke(draw, (template_w-w)/2, cy, line, fb, (50,50,50), (255,255,255), 2)
            cy += line_height

        buf = io.BytesIO()
        template = template.convert("RGB")
        template.save(buf, format="JPEG", quality=90, optimize=True)
        final_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        torch.cuda.empty_cache()

        # 반환값도 정리
        return {"profile_text": '\n'.join(info_lines + [generated_story]), "profile_image_base64": final_b64}

    except Exception as e:
        print(f"🚨 Adoption Profile Error: {e}")
        raise HTTPException(status_code=422, detail="Unsupported Image Format or Corrupt File.")

# =========================================================
# 3. 스튜디오 프로필 (누끼 + 중앙 정렬 + HEIC 지원)
# =========================================================
@app.post("/api/v1/generate-studio-profile", response_model=dict)
async def generate_studio_profile(
    image: UploadFile = File(...),
    bg_color: str = Form("#FFD1DC") 
):
    if "upsampler" not in models or "remover" not in models: 
        raise HTTPException(status_code=503, detail="Model Loading")

    try:
        contents = await image.read()
        if len(contents) == 0: raise ValueError("Uploaded file is empty.")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        
        w, h = img.size
        
        TARGET_SIZE = 1280
        if max(w, h) > TARGET_SIZE:
            scale = TARGET_SIZE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        if max(w, h) < 1000:
            cv2_img = pil_to_cv2(img)
            output, _ = models["upsampler"].enhance(cv2_img, outscale=4)
            img = cv2_to_pil(output)
            img = resize_image_if_too_large(img, max_dim=1500)

        no_bg = remove(
            img, 
            session=models["remover"], 
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )

        bbox = no_bg.getbbox() 
        if bbox:
            subject_only = no_bg.crop(bbox) 
        else:
            subject_only = no_bg 

        TARGET_W, TARGET_H = 1080, 1350
        try:
            color_rgb = ImageColor.getrgb(bg_color)
        except:
            color_rgb = (255, 240, 245)
        final_canvas = Image.new("RGB", (TARGET_W, TARGET_H), color_rgb)

        MAX_SUB_W = int(TARGET_W * 0.9)
        MAX_SUB_H = int(TARGET_H * 0.9)

        s_w, s_h = subject_only.size
        scale_w = MAX_SUB_W / s_w
        scale_h = MAX_SUB_H / s_h
        scale_factor = min(scale_w, scale_h) 

        new_s_w = int(s_w * scale_factor)
        new_s_h = int(s_h * scale_factor)
        resized_subject = subject_only.resize((new_s_w, new_s_h), Image.LANCZOS)

        paste_x = (TARGET_W - new_s_w) // 2
        paste_y = (TARGET_H - new_s_h) // 2
        final_canvas.paste(resized_subject, (paste_x, paste_y), resized_subject)

        buf = io.BytesIO()
        final_canvas.save(buf, format="JPEG", quality=90, optimize=True)
        final_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        torch.cuda.empty_cache()

        return {"profile_image_base64": final_b64, "message": "성공"}

    except Exception as e:
        print(f"🚨 Studio Profile Error: {e}")
        return {"profile_image_base64": "", "message": "Error"}
