import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from elasticsearch7 import Elasticsearch
from datetime import datetime, timedelta
from pytz import timezone
import boto3
from botocore.client import Config
import pickle

# 1. 테스트용 데이터 생성 (훈련 데이터를 위한 가상 데이터)
np.random.seed(42)
player_ids = [f'player_{i}' for i in range(1, 101)]  # 100명의 플레이어

# 각 플레이어의 액션 카운트 및 액션 타입 생성
action_counts = np.random.randint(1, 50, size=100)
action_types = [np.random.choice(['move', 'attack', 'defend'], size=np.random.randint(1, 5), replace=True).tolist() for _ in range(100)]

# 마지막 액션 시점 생성 (현재 시점으로부터 최대 30일 전, 랜덤하게)
current_time = datetime.now()
last_action_timestamps = [current_time - timedelta(days=np.random.randint(0, 30)) for _ in range(100)]

# 이탈 여부 레이블링 (예: 최근 7일 동안 활동이 없으면 이탈로 간주, 랜덤성 추가)
recency = [(current_time - ts).days for ts in last_action_timestamps]
churn = [1 if r > 7 else 0 for r in recency]

# 일부 데이터를 랜덤하게 뒤섞어 정확한 분류가 어렵게 함으로써 랜덤성 부여
random_noise = np.random.binomial(1, 0.1, size=100)  # 10% 확률로 레이블을 뒤바꿈
churn = [1 - val if noise == 1 else val for val, noise in zip(churn, random_noise)]

# DataFrame 생성
data = pd.DataFrame({
    'playerId': player_ids,
    'action_count': action_counts,
    'last_action': last_action_timestamps,
    'recency': recency,
    'churn': churn
})

# 각 액션 타입별 카운트 계산 (dummy data로 생성)
action_type_counts = pd.DataFrame([{action: np.random.randint(0, 10) for action in ['move', 'attack', 'defend']} for _ in range(100)])
action_type_counts['playerId'] = player_ids

# 특징 데이터 결합
features = data.merge(action_type_counts, on='playerId')

# 모델 입력 변수와 타겟 변수 정의
X = features.drop(columns=['playerId', 'churn', 'last_action'])
y = features['churn']

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 모델 정의 및 학습
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. 모델 평가
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# 7. 모델 저장 (MinIO에 저장)
# MinIO 설정
minio_endpoint = 'http://192.168.49.2:31433'
access_key = 'minioadmin'
secret_key = 'minioadmin'

s3 = boto3.client('s3',
                  endpoint_url=minio_endpoint,
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key,
                  config=Config(signature_version='s3v4'),
                  region_name='us-east-1')

# 모델을 로컬에 저장
with open('/tmp/saved_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# MinIO에 업로드
bucket_name = 'ai-bucket'
model_file_path = '/tmp/saved_model.pkl'
model_object_name = 'churn_model.pkl'

# 버킷이 존재하지 않으면 생성
try:
    s3.head_bucket(Bucket=bucket_name)
except Exception as e:
    s3.create_bucket(Bucket=bucket_name)

# 모델 파일 업로드
s3.upload_file(model_file_path, bucket_name, model_object_name)

print("Model saved and uploaded to MinIO successfully.")