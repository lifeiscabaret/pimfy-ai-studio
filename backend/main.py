from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os  # .env용
from dotenv import load_dotenv  # .env용
import torch
# (!!) Image-to-Image 파이프라인으로 변경
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from io import BytesIO
import asyncio
import databases
import sqlalchemy
import requests # 추가 (이미지다운로드 로직)
from PIL import Image # <-- 이미지 처리용
from rembg import remove # <-- 배경 제거용

# --- 1. Cafe24 DB (MySQL) 설정 (보안 강화!) ---
load_dotenv() # .env 파일 로드
DATABASE_URL = os.getenv("DATABASE_URL") # 환경 변수에서 읽기
# IMAGE_BASE_PATH는 .env 파일에서 읽어옴 (올바른 웹 경로: /inday_fileinfo/img)
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/inday_fileinfo/img") # 기본값 설정

if not DATABASE_URL:
    print("🚨 치명적 에러: .env 파일에 DATABASE_URL이 설정되지 않았습니다! 서버를 시작할 수 없습니다.")
    # 실제 운영 시에는 여기서 에러를 발생시키거나 기본값 설정 필요
    raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# 'homeprotection' 테이블 정의
dogs_table = sqlalchemy.Table(
    "homeprotection",
    metadata,
    sqlalchemy.Column("uid", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("subject", sqlalchemy.String(250)),      # 유기견 이름
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),      # 이미지 파일 (파일명!)
    sqlalchemy.Column("addinfo03", sqlalchemy.String(10)),       # 성별
    sqlalchemy.Column("addinfo04", sqlalchemy.String(10)),       # 중성화 여부
    sqlalchemy.Column("addinfo05", sqlalchemy.String(10)),       # 출생 시기 (나이)
    sqlalchemy.Column("addinfo07", sqlalchemy.String(10)),       # 몸무게
    sqlalchemy.Column("addinfo08", sqlalchemy.Text),             # 성격 태그
    sqlalchemy.Column("addinfo09", sqlalchemy.Text),             # 구조 사연
    sqlalchemy.Column("addinfo10", sqlalchemy.Text),             # 성격 및 특징
    sqlalchemy.Column("addinfo11", sqlalchemy.Text),             # 기타 사항
    sqlalchemy.Column("addinfo19", sqlalchemy.String(250)),      # 병력 사항
)

# --- 2. Pydantic 모델 정의 ---
class Dog(BaseModel):
    uid: int
    subject: str
    s_pic01: str
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
    profile_image_base64: str

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

    # (!!) Image-to-Image 파이프라인 로드 (SD 1.5 기반)
    print("Loading Stable Diffusion Image-to-Image pipeline...")
    models["image_pipe"] = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)

    # (!!) 5.8B 모델 (GPU 메모리 최적화)
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


# --- 7. AI 프로필 생성 API (Image-to-Image + 보안) ---
@app.post("/api/v1/generate-real-profile", response_model=ProfileResponse)
async def generate_real_profile(request: RealProfileRequest):
    if "image_pipe" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았거나 준비 중입니다. 잠시 후 다시 시도해 주세요.")

    dog_dict = await get_dog_details(request.dog_uid)
    dog = Dog(**dog_dict) # Pydantic 모델로 변환

    original_rgb_image_base64 = None # 원본 이미지의 Base64를 저장할 변수

    # --- 이미지 처리 (HTTP 다운로드 방식) ---
    img_str = "Error processing image" # 기본값
    try:
        # .env에서 URL 정보 읽기
        SITE_BASE_URL = "https://www.pimfyvirus.com" # (하드코딩)
        IMAGE_WEB_PATH = IMAGE_BASE_PATH # .env에서 "/inday_fileinfo/img"를 읽어옴 (따옴표 없이!)

        if not dog.s_pic01:
            raise ValueError("DB에 이미지 파일명(s_pic01)이 없습니다.")

        # 'https://.../inday_fileinfo/img/filename.jpg' URL 조합
        image_url = f"{SITE_BASE_URL}{IMAGE_WEB_PATH}/{dog.s_pic01}"

        print(f"이미지 다운로드 시도: {image_url}")

        # Requests로 이미지 다운로드
        response = requests.get(image_url, stream=True)
        response.raise_for_status() # 404 등 에러가 나면 여기서 멈춤

        print("이미지 다운로드 성공, PIL로 여는 중...")

        # 다운로드한 이미지를 PIL로 열기
        input_image = Image.open(response.raw).convert("RGB")

        # --- (새로운 기능!) 원본 이미지를 Base64로 저장해두기 ---
        buffered_original = BytesIO()
        input_image.save(buffered_original, format="PNG")
        original_rgb_image_base64 = base64.b64encode(buffered_original.getvalue()).decode("utf-8")
        # ----------------------------------------------------

        print("배경 제거 시작...")
        # `only_mask=True`를 사용해 마스크만 추출 후 원본 이미지와 합성하는 방식도 고려 가능
        output_image = remove(input_image, alpha_matting=True)
        print("배경 제거 완료.")

        output_image = output_image.resize((512, 512))
        # 배경 제거 후 투명한 부분을 흰색으로 채움 (Stable Diffusion 입력용)
        rgb_image_for_sd = Image.new("RGB", output_image.size, (255, 255, 255))
        rgb_image_for_sd.paste(output_image, mask=output_image.split()[3])

        # --- (✨ 중요!) 긍정 & 부정 프롬프트 생성 ---
        dog_name = dog.subject if dog.subject else "this dog"
        personality_tags = f", {dog.addinfo08}" if dog.addinfo08 else ""

        # 긍정 프롬프트: 원하는 결과물 (스튜디오 품질, 선명함 강조)
        prompt_image = f"""
        professional studio portrait photo of {dog_name}{personality_tags}, centered, medium shot view, high resolution,
        masterpiece, best quality, sharp focus, highly detailed fur texture, natural lighting, photo-realistic, cinematic quality,
        plain light gray background
        """.strip().replace("\n", " ") # 줄바꿈 제거

        # 부정 프롬프트: 피해야 할 요소들 (흐릿함, 저퀄리티, 배경 요소 제거)
        negative_prompt = """
        blurry, low quality, worst quality, unclear, unfocused, distorted, disfigured, ugly, deformed, bad anatomy, extra limbs, missing limbs, mutated hands, mutation,
        cartoon, drawing, sketch, illustration, painting, anime, 3d render, illustration, drawing, painting, sketch, cartoon, anime, manga, doll, toy, plastic, fake
        watermark, text, signature, words, letters, noise, grain, artifacts, compression artifacts, jpeg artifacts, overexposed, underexposed, bad lighting, multiple dogs, human, person, hands, feet,
        cage, bars, leash, harness, chain, wires, fence, outdoor, nature, grass, trees, buildings, furniture, messy background, cluttered background
        """.strip().replace("\n", " ") # 줄바꿈 제거

        print(f"Using Image Prompt: {prompt_image}")
        print(f"Using Negative Prompt: {negative_prompt}")

        print("이미지 개선 시작...")
        # (!!!) strength 값 조절: 특이한 이미지의 경우 더 낮게 (0.3~0.4) 설정하여 원본 형태 유지
        enhanced_image = models["image_pipe"](
            prompt=prompt_image,
            negative_prompt=negative_prompt,
            image=rgb_image_for_sd,
            strength=0.35, # <-- ★★★ 이 값을 0.3~0.45 정도로 낮춰서 테스트해보세요!
            guidance_scale=8.0
        ).images[0]
        print("이미지 개선 완료.")

        buffered = BytesIO()
        enhanced_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        # (!!) 이미지 생성 실패 시, 원본 이미지를 그대로 반환하도록 처리
        if original_rgb_image_base64:
            img_str = original_rgb_image_base64
            print("AI 이미지 생성 실패, 원본 이미지를 대신 반환합니다.")
        else:
            img_str = "Error: Could not process or retrieve image." # 원본도 없는 경우

    # --- 텍스트 생성 ---
    # (✨ 수정 제안) 데이터 전처리 예시 (간단 버전)
    def clean_text(text):
        if not text: return ""
        # 간단하게 HTML 태그 제거 (더 강력한 방법 필요할 수 있음)
        import re
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
    print(f"DEBUG: Text Prompt:\n{prompt_text}") # <-- 디버깅용: 실제 프롬프트 확인

    generated_text = "Error generating text"
    try:
        inputs = models["tokenizer"](prompt_text, return_tensors="pt").to(models["text_model"].device)
        output_sequences = models["text_model"].generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            temperature=0.6,  # <--- ✨ 0.7에서 0.6 정도로 낮춰서 무작위성 감소
            repetition_penalty=1.2,
            early_stopping=True
        )
        decoded_text = models["tokenizer"].decode(output_sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = decoded_text.split("소개글:")[-1].strip()
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")


    return {
        "profile_text": generated_text,
        "profile_image_base64": img_str
    }