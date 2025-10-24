from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import torch
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from io import BytesIO
import asyncio
import databases
import sqlalchemy

# --- 1. Cafe24 DB (MySQL) 설정 ---
load_dotenv() # <-- .env 파일 로드하는 코드 추가
DATABASE_URL = os.getenv("DATABASE_URL")

# (!!) DATABASE_URL이 제대로 로드되었는지 확인 (선택 사항)
if not DATABASE_URL:
    print("🚨 에러: .env 파일에 DATABASE_URL이 설정되지 않았습니다!")
    # 또는 raise Exception(...) 등으로 에러 처리

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# (!!) '진짜' 유기견 공고 테이블 'homeprotection'
dogs_table = sqlalchemy.Table(
    "homeprotection",
    metadata,
    sqlalchemy.Column("uid", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("subject", sqlalchemy.String(250)),      # 유기견 이름
    sqlalchemy.Column("s_pic01", sqlalchemy.String(150)),      # 이미지 파일
    sqlalchemy.Column("addinfo03", sqlalchemy.String(10)),       # 성별
    sqlalchemy.Column("addinfo04", sqlalchemy.String(10)),       # (!!) 중성화 여부
    sqlalchemy.Column("addinfo05", sqlalchemy.String(10)),       # 출생 시기 (나이)
    sqlalchemy.Column("addinfo07", sqlalchemy.String(10)),       # (!!) 몸무게
    sqlalchemy.Column("addinfo08", sqlalchemy.Text),             # (!!) 성격 태그
    sqlalchemy.Column("addinfo09", sqlalchemy.Text),             # 구조 사연
    sqlalchemy.Column("addinfo10", sqlalchemy.Text),             # 성격 및 특징
    sqlalchemy.Column("addinfo11", sqlalchemy.Text),             # (!!) 기타 사항
    sqlalchemy.Column("addinfo19", sqlalchemy.String(250)),      # (!!) 병력 사항
)

# --- 2. Pydantic 모델 정의 (API 입출력 형식) ---
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

# --- 3. (!!) FastAPI 앱 & AI 모델 변수 선언 ---
models = {} 
app = FastAPI() # (!!) 'app'을 여기서 먼저 정의합니다.

# --- 4. '진짜' AI 모델 로딩 (서버 시작 시) ---
@app.on_event("startup")
def load_models_and_db():
    print("Cafe24 DB 연결 준비... (각 API 요청 시 연결)")

    print("AI 모델 로딩을 시작합니다... (시간이 몇 분 정도 걸릴 수 있습니다)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    models["image_pipe"] = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    
    models["tokenizer"] = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    models["text_model"] = AutoModelForCausalLM.from_pretrained(
    "beomi/KoAlpaca-Polyglot-5.8B", # <--- 이 부분!
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
    return await db.fetch_all(query)

@app.get("/api/dogs/{dog_uid}", response_model=Dog)
async def get_dog_details(dog_uid: int):
    db = await get_db_connection()
    query = dogs_table.select().where(dogs_table.c.uid == dog_uid)
    dog = await db.fetch_one(query)
    if not dog:
        raise HTTPException(status_code=404, detail="해당 ID의 강아지 정보를 찾을 수 없습니다.")
    return dog


# --- 7. AI 프로필 생성 API ---
@app.post("/api/v1/generate-real-profile", response_model=ProfileResponse)
async def generate_real_profile(request: RealProfileRequest):
    if "image_pipe" not in models or "text_model" not in models:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았거나 준비 중입니다. 잠시 후 다시 시도해 주세요.")

    dog = await get_dog_details(request.dog_uid)

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
    
    inputs = models["tokenizer"](prompt_text, return_tensors="pt").to(models["text_model"].device)
    
    # (!!) (!!) (!!) 에러 수정! (v4)
    # Ko-Alpaca (GPT-NeoX) 모델은 'token_type_ids'를 사용하지 않는데 넣어서 충돌발생.
    # inputs 대신, 필요한 'input_ids'와 'attention_mask'만 명시적으로 전달합니다.
    output_sequences = models["text_model"].generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_new_tokens=300, 
        temperature=0.7, 
        repetition_penalty=1.2
    )
    
    generated_text = models["tokenizer"].decode(output_sequences[0], skip_special_tokens=True)
    
    prompt_image = f"A high-resolution, heartwarming studio photo of a cute dog named {dog.subject}, looking at the camera"
    image = models["image_pipe"](prompt=prompt_image).images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "profile_text": generated_text.split("소개글:")[-1].strip(),
        "profile_image_base64": img_str
    }