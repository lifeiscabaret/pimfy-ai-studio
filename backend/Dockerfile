FROM nvcr.io/nvidia/tensorrt:23.09-py3

# 1. 시스템 패키지 설치
RUN apt-get update -o Acquire::AllowInsecureRepositories=true \
                   -o Acquire::AllowDowngradeToInsecureRepositories=true \
                   -o Acquire::AllowInvalidAuthentications=true \
    || true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ⭐️ 2. 파이토치(PyTorch) CUDA 버전 강제 설치
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 나머지 라이브러리 설치
RUN pip install --no-cache-dir git+https://github.com/xinntao/BasicSR.git && \
    pip install --no-cache-dir -r requirements.txt

# 4. 모델 파일 다운로드 (.pth 원본 사용)
RUN mkdir -p /app/esrgan/ && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/esrgan/RealESRGAN_x4plus.pth

# 5. 소스 복사
COPY main.py .
COPY KyoboHandwriting2021sjy.otf /app/
COPY NanumGothic-Regular.ttf /app/
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
