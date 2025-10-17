# Python FastAPI를 사용하여 백엔드 서버를 구성할 예정.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- Pydantic 모델 정의 ---
class DogProfile(BaseModel):
    id: int
    name: str
    breed: str
    age: int
    story: str
    imageUrl: str
    shelter: str

# --- 임시 데이터베이스 (Mock Data) ---
# 실제 핌피바이러스 DB 연동 전, 테스트를 위한 가상 데이터입니다.
MOCK_DB: List[DogProfile] = [
    DogProfile(id=1, name="해피", breed="믹스견", age=2, story="사람을 정말 좋아하는 애교쟁이 해피입니다. 새로운 가족을 기다려요!", imageUrl="https://placehold.co/400x300/FFCAD4/333?text=Happy", shelter="핌피 보호소 (서울)"),
    DogProfile(id=2, name="콩이", breed="푸들", age=5, story="조용하고 얌전한 성격의 콩이는 든든한 가족이 되어줄 거예요.", imageUrl="https://placehold.co/400x300/FFF5D1/333?text=Kong", shelter="사랑 쉼터 (부산)"),
    DogProfile(id=3, name="별이", breed="진돗개 믹스", age=1, story="겁이 많지만 마음을 열면 한없이 다정한 별이의 평생 가족을 찾습니다.", imageUrl="https://placehold.co/400x300/C8F3E0/333?text=Star", shelter="희망 보호소 (대구)"),
    DogProfile(id=4, name="해피투", breed="골든 리트리버", age=3, story="에너지가 넘치고 똑똑한 해피투! 함께 산책할 가족을 구해요.", imageUrl="https://placehold.co/400x300/FF7A4D/333?text=Happy2", shelter="핌피 보호소 (서울)"),
]

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- CORS 미들웨어 설정 ---
# 프론트엔드(http://localhost:3000)에서 백엔드(http://localhost:8000)로
# API를 요청할 수 있도록 허용해주는 설정임.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API 엔드포인트 정의 ---
@app.get("/")
def read_root():
    """ 서버 상태를 확인하기 위한 기본 엔드포인트 """
    return {"message": "Pimfy Profile Generator API"}


@app.get("/api/dogs/search", response_model=List[DogProfile])
def search_dogs(name: Optional[str] = None):
    """
    유기견 이름을 검색하는 API 엔드포인트
    - 요청 예시: /api/dogs/search?name=해피
    - 이름이 주어지면 해당 이름이 포함된 강아지 목록을 반환합니다.
    - 이름이 없으면 전체 목록을 반환합니다.
    """
    if not name:
        return MOCK_DB

    # 대소문자 구분 없이 검색
    search_results = [dog for dog in MOCK_DB if name.lower() in dog.name.lower()]
    return search_results
