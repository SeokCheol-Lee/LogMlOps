FROM python:3.8-slim

# 필요한 패키지 설치
RUN pip install pandas scikit-learn elasticsearch7 boto3 joblib

# 작업 디렉토리 설정
WORKDIR /app

# 스크립트 복사
COPY data_processing.py /app/
COPY model_training.py /app/

# 엔트리포인트 설정 (필요에 따라)
ENTRYPOINT ["python", "model_training.py"]
