# 1. [교체] 문제 많은 NVIDIA 이미지 버리고, 공식 PyTorch 이미지 사용
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 2. [설치] 키 에러? 권한 문제? 이제 안 생깁니다.
# 건강한 이미지라서 그냥 설치하면 됩니다.
RUN apt-get update && apt-get install -y --no-install-recommends     libgl1-mesa-glx     libglib2.0-0     git     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 3. [그대로 유지] 아까 받아둔 파일 활용 (보안 O, 속도 O)
COPY BasicSR ./BasicSR
# PyTorch는 이미 깔려있으니 건너뛰고 나머지 설치
RUN pip install --no-cache-dir ./BasicSR && pip install --no-cache-dir -r requirements.txt

# 4. 모델 및 실행 파일 복사
COPY RealESRGAN_x4plus.pth /app/esrgan/RealESRGAN_x4plus.pth
COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
