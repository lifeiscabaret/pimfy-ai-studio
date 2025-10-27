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
from PIL import Image # <-- 이미지 처리용
from rembg import remove # <-- 배경 제거용

# --- 1. Cafe24 DB (MySQL) 설정 (보안 강화!) ---
load_dotenv() # .env 파일 로드
DATABASE_URL = os.getenv("DATABASE_URL") # 환경 변수에서 읽기
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "/www/inday_fileinfo/img") # .env에서 읽기 (없을 경우 기본값 사용)

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
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),      # 이미지 파일 (경로!)
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
    # SDXL Image-to-Image는 다른 모델 ID를 사용할 수 있음 (나중에 업그레이드 고려)
    models["image_pipe"] = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", # <-- 우선 SD 1.5 img2img 사용
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
    # Pydantic v1 호환성을 위해 RowProxy를 dict로 변환 (FastAPI 구버전 등에서 필요할 수 있음)
    results = await db.fetch_all(query)
    return [dict(row) for row in results]


@app.get("/api/dogs/{dog_uid}", response_model=Dog)
async def get_dog_details(dog_uid: int):
    db = await get_db_connection()
    query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog = await db.fetch_one(query)
    if not dog:
        raise HTTPException(status_code=404, detail="해당 ID의 강아지 정보를 찾을 수 없습니다.")
    # Pydantic v1 호환성을 위해 RowProxy를 dict로 변환
    return dict(dog)


# --- 7. AI 프로필 생성 API (Image-to-Image + 보안) ---
@app.post("/api/v1/generate-real-profile", response_model=ProfileResponse)
async def generate_real_profile(request: RealProfileRequest):
    if "image_pipe" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았거나 준비 중입니다. 잠시 후 다시 시도해 주세요.")

    dog_dict = await get_dog_details(request.dog_uid)
    dog = Dog(**dog_dict) # Pydantic 모델로 변환

    # --- 이미지 처리 ---
    img_str = "Error processing image" # 기본값
    try:
        image_path = dog.s_pic01
# (!!) Cafe24 서버의 '절대 파일 경로'로 이미지를 직접 열어야 함.
        image_folder_path = IMAGE_BASE_PATH
        
        if not dog.s_pic01:
            raise ValueError("DB에 이미지 경로(s_pic01)가 없습니다.")

        # os.path.join을 사용해 절대 경로 조합 (예: /www/inday_fileinfo/img/filename.jpg)
        full_file_path = os.path.join(image_folder_path, dog.s_pic01)
        
        print(f"서버 내부 경로에서 이미지 여는 중: {full_file_path}")
        
        if not os.path.exists(full_file_path):
            print(f"경고: 파일을 찾을 수 없습니다! {full_file_path}")
            raise HTTPException(status_code=404, detail=f"Image file not found at path: {full_file_path}")
            
        # HTTP 요청 대신 파일 시스템에서 직접 이미지를 열고
        input_image = Image.open(full_file_path).convert("RGB")

        print("배경 제거 시작...")
        # (!!) alpha_matting=True 옵션 추가 (선택적, 더 나은 품질 위해)
        output_image = remove(input_image, alpha_matting=True) # RGBA
        print("배경 제거 완료.")

        # Image-to-Image 입력 준비 (배경 제거된 이미지 사용)
        output_image = output_image.resize((512, 512)) # SD 1.5 기본 해상도
        # RGBA -> RGB 변환 (흰색 배경)
        rgb_image = Image.new("RGB", output_image.size, (255, 255, 255))
        rgb_image.paste(output_image, mask=output_image.split()[3])

        print("이미지 개선 시작...")
        prompt_image = f"high-quality professional studio photo of this cute dog named {dog.subject}, realistic, masterpiece, best quality, centered, plain light gray background" # 배경색 지정
        # strength 낮추면 원본 유지력 상승, 높이면 AI 창의성 증가
        enhanced_image = models["image_pipe"](prompt=prompt_image, image=rgb_image, strength=0.6, guidance_scale=7.5).images[0]
        print("이미지 개선 완료.")

        buffered = BytesIO()
        enhanced_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        # 이미지 처리 실패 시에도 텍스트는 생성하도록 계속 진행

    # --- 텍스트 생성 ---
    prompt_text = f"""
    # MISSION (임무)
    당신은 국내 최고의 동물 구조 전문 카피라이터입니다.
    당신의 유일한 임무는, 아래 [견종 정보]를 가진 유기견에게 '평생 가족'을 찾아주는 것입니다.
    이 아이가 아니면 안 되겠다는 '운명적인 끌림'을 느끼게 만드는, 감성적이고 임팩트 있는 프로필을 작성해 주세요.

    # INSTRUCTIONS (작성 지침)
    1.  **내용 충실:** [견종 정보]에 있는 사실만을 바탕으로 작성해야 합니다.
    2.  **단점 승화:** 아이의 아픈 '사연'이나 '병력'은 '극복과 희망'으로, '특징'은 '매력'으로 승화시켜 주세요.
    3.  **감성 자극:** 독자의 마음을 움직이고, 이 아이와 함께하는 미래를 그리고 싶다는 '핵심 욕구'를 자극해 주세요.
    4.  **어조:** 따뜻하고 다정한 관찰자 시점으로 작성해 주세요. (1인칭 X)

    # 견종 정보 (Dog's Data)
    - 이름: {dog.subject}
    - 성별: {dog.addinfo03}
    - 나이(출생시기): {dog.addinfo05}
    - 몸무게: {dog.addinfo07}kg
    - 중성화: {dog.addinfo04}
    - 성격 태그: {dog.addinfo08}
    - 성격 및 특징: {dog.addinfo10}
    - 구조 사연: {dog.addinfo09}
    - 병력 사항: {dog.addinfo19}
    - 기타 사항: {dog.addinfo11}
    ---
    # PROFILE (프로필 작성)
    소개글:
    """

    generated_text = "Error generating text" # 기본값
    try:
        inputs = models["tokenizer"](prompt_text, return_tensors="pt").to(models["text_model"].device)
        output_sequences = models["text_model"].generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=300,
            temperature=0.7,
            repetition_penalty=1.2,
            # (!!) early_stopping=True 추가 (생성 완료 시 빠르게 종료)
            early_stopping=True
        )
        # (!!) 텍스트 깨짐 해결 시도 (v6 유지)
        decoded_text = models["tokenizer"].decode(output_sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_text = decoded_text.split("소개글:")[-1].strip()
        # (!!) 추가: 만약 그래도 깨진다면 UTF-8 강제 인코딩/디코딩 시도 (하지만 보통 decode가 처리함)
        # generated_text = generated_text.encode('latin1').decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")


    return {
        "profile_text": generated_text,
        "profile_image_base64": img_str
    }
