# 1. 베이스 이미지 선택 (NVIDIA CUDA 런타임이 포함된 이미지)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. 컨테이너 내부의 작업 폴더를 '/app'으로 지정.
WORKDIR /app

# 3. 파이썬과 pip(라이브러리 설치 도구) 설치.
RUN apt-get update && apt-get install -y python3-pip

# 4. 재료 목록(requirements.txt)을 먼저 컨테이너 안으로 복사하고
COPY requirements.txt .

# 5. 재료 목록에 있는 모든 파이썬 라이브러리 설치.
RUN pip install --no-cache-dir -r requirements.txt
# (!!) .env 파일을 읽기 위한 python-dotenv 라이브러리 추가.
RUN pip install python-dotenv

# 6. 현재 폴더의 나머지 모든 소스 코드(main.py 등)를 복사.
COPY . .

# 7. 외부에서 컨테이너의 8000번 포트로 접속.
EXPOSE 8000

# 8. 컨테이너가 시작될 때 최종적으로 실행할 명령어를 지정.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]