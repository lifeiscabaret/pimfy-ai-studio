# 1. [교체] NVIDIA 이미지 ->  공식 PyTorch 이미지 사용
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 2. [설치]
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1-mesa-glx     libglib2.0-0     git     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./backend/requirements.txt .

# 3. [BasicSR]
COPY BasicSR ./BasicSR
# PyTorch는 이미 깔려있으니 건너뛰고 나머지 설치
RUN pip install --no-cache-dir ./BasicSR && pip install --no-cache-dir -r requirements.txt

# 4. 모델 및 실행 파일 복사
COPY fonts/ /app/fonts/
COPY RealESRGAN_x4plus.pth /app/esrgan/RealESRGAN_x4plus.pth
COPY ./backend/main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
