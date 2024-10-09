# 베이스 이미지 설정 (Python 포함, Debian Bullseye 사용)
FROM python:3.9-bullseye

# 1. 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    libgfortran5 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. pip, setuptools, wheel 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 복사
COPY requirements.txt /app/requirements.txt

# 5. NumPy 먼저 설치하여 버전 고정
RUN pip install --no-cache-dir numpy==1.21.6

# 6. 나머지 패키지 설치 (NumPy 버전 업그레이드 방지)
RUN pip install --no-cache-dir -r requirements.txt

# 7. 패키지 목록 출력 (디버깅용)
RUN pip list

# 8. NumPy 설치 경로 확인 (디버깅용)
RUN python -c "import numpy as np; print('NumPy is installed at:', np.__file__)"

# 9. 패키지 설치 확인 (디버깅용)
RUN python -c "import numpy as np; print('NumPy version:', np.__version__); \
               import pandas as pd; import sklearn; import joblib; import minio; import elasticsearch; \
               print('All packages imported successfully.')"

# 10. 학습 스크립트 복사
COPY es_log_train_model.py /app/es_log_train_model.py

# 11. 필요한 디렉토리 생성
RUN mkdir -p /data

# 12. 스크립트 실행
CMD ["python", "es_log_train_model.py"]
