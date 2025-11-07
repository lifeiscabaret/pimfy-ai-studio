FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 1. cuDNN 설치, 파이썬 기본 의존성, Git 설치, wget을 단일 RUN 명령어로 통합
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 컨테이너 내부의 작업 폴더를 '/app'으로 지정.
WORKDIR /app

# 3. Real-ESRGAN 모델 파일 다운로드 및 독립된 경로에 복사 (볼륨 충돌 방지)
RUN mkdir -p /app/esrgan/ && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/esrgan/RealESRGAN_x4plus.pth

# 4. NVIDIA 라이브러리 경로를 환경 변수에 명시적으로 추가
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 5. 재료 목록(requirements.txt)을 먼저 컨테이너 안으로 복사
COPY requirements.txt .

# 6. 파이썬 라이브러리 설치.
RUN pip install --no-cache-dir git+https://github.com/xinntao/BasicSR.git && \
    pip install --no-cache-dir -r requirements.txt

# --- 폰트 파일을 /app/ 경로로 복사 ---
COPY NanumGothic-Bold.ttf /app/
COPY NanumGothic-Regular.ttf /app/
# ---------------------------------------------

# 7. 현재 폴더의 나머지 모든 소스 코드(main.py 등)를 복사.
COPY . .

# 8. libcudnn8과 libcudnn9 간의 링크 연결 (rembg의 GPU 가속을 위한 최종 시도)
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 || true

# 9. 외부에서 컨테이너의 8000번 포트로 접속.
EXPOSE 8000

# 10. 컨테이너가 시작될 때 최종적으로 실행할 명령어를 지정.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
