# 1. 베이스 이미지 선택 ...
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. 컨테이너 내부의 작업 폴더를 '/app'으로 지정.
WORKDIR /app

# (✨ 추가!) NVIDIA 라이브러리 경로를 환경 변수에 명시적으로 추가
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 3. 파이썬과 pip, (✨cv2 의존성 5종 세트!) 설치.
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 4. 재료 목록(requirements.txt)을 먼저 컨테이너 안으로 복사
COPY requirements.txt .

# 5. 재료 목록에 있는 모든 파이썬 라이브러리 설치.
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# (✨ 수정!) 버그가 해결된 BasicSR를 '가장 먼저' 강제로 설치
RUN pip install --no-cache-dir git+https://github.com/xinntao/BasicSR.git

# (✨ 수정!) BasicSR이 이미 설치됐으니, 나머지 재료들만 설치
RUN pip install --no-cache-dir -r requirements.txt

# --- 폰트 파일을 /app/ 경로로 복사 ---
COPY NanumGothic-Bold.ttf /app/
COPY NanumGothic-Regular.ttf /app/
# ---------------------------------------------

# 6. 현재 폴더의 나머지 모든 소스 코드(main.py 등)를 복사.
# (폰트와 requirements.txt가 이미 복사되었으므로, 나머지 코드만 복사됩니다)
COPY . .

# 7. 외부에서 컨테이너의 8000번 포트로 접속.
EXPOSE 8000

# 8. 컨테이너가 시작될 때 최종적으로 실행할 명령어를 지정.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
