# 1. Python 3.10 Slim 버전 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 파일 복사
COPY requirements.txt .
COPY . .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. FastAPI 앱 실행 (chatbot.py에서 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
