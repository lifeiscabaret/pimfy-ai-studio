# 기본 이미지: nvcr.io/nvidia/tensorrt:23.09-py3 (CUDA 11.8 기반, TensorRT 포함)
FROM nvcr.io/nvidia/tensorrt:23.09-py3

# 1. 시스템 기본 의존성 설치 (git, wget 및 이미지 처리 라이브러리 추가)
# libgl/libjpeg/zlib 등의 의존성 추가
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 컨테이너 내부의 작업 폴더를 '/app'으로 지정.
WORKDIR /app

# 3. Real-ESRGAN 모델 파일 다운로드 및 독립된 경로에 복사
RUN mkdir -p /app/esrgan/ && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/esrgan/RealESRGAN_x4plus.pth

# 4. 재료 목록(requirements.txt)을 먼저 컨테이너 안으로 복사
COPY requirements.txt .

# 5. 파이썬 라이브러리 설치.
RUN pip install --no-cache-dir git+https://github.com/xinntao/BasicSR.git && \
    pip install --no-cache-dir -r requirements.txt

# --- 폰트 파일을 /app/ 경로로 복사 ---
COPY NanumGothic-Bold.ttf /app/
COPY NanumGothic-Regular.ttf /app/
# ---------------------------------------------

# 6. 현재 폴더의 나머지 모든 소스 코드(main.py 등)를 복사.
COPY . .

# 7. 외부에서 컨테이너의 8000번 포트로 접속.
EXPOSE 8000

# 8. 컨테이너가 시작될 때 최종적으로 실행할 명령어를 지정.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
