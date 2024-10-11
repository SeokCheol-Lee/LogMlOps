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

# 1. Elasticsearch에서 데이터 로드
es = Elasticsearch(hosts=["192.168.49.2:32377"])

# 예시로 최근 30일 간의 데이터를 가져옵니다.
utc_time = datetime.now(timezone('UTC'))
utc_plus_9 = timezone('Asia/Seoul')
end_time = utc_time.astimezone(utc_plus_9)
start_time = end_time - timedelta(days=30)

# 시간 범위를 ISO 포맷으로 변환
start_time_iso = start_time.isoformat()
end_time_iso = end_time.isoformat()

# Elasticsearch 쿼리 정의
query_body = {
    "range": {
        "@timestamp": {
            "gte": start_time_iso,
            "lte": end_time_iso
        }
    }
}

# 데이터 검색 (query와 size 파라미터로 수정)
res = es.search(index="logs", query=query_body, scroll='2m', size=10000)
scroll_id = res['_scroll_id']
hits = res['hits']['hits']

# 모든 데이터 수집
all_hits = []
all_hits.extend(hits)

while len(hits) > 0:
    res = es.scroll(scroll_id=scroll_id, scroll='2m')
    hits = res['hits']['hits']
    if not hits:
        break
    all_hits.extend(hits)

# DataFrame으로 변환
data_list = []
for hit in all_hits:
    source = hit['_source']
    data_list.append({
        'playerId': source['playerId'],
        'actionType': source['actionType'],
        '@timestamp': source['@timestamp']
    })

data = pd.DataFrame(data_list)

# 2. 데이터 전처리

# timestamp를 datetime 형식으로 변환
data['@timestamp'] = pd.to_datetime(data['@timestamp'])

# 액션 카운트 계산
action_counts = data.groupby('playerId').size().reset_index(name='action_count')

# 각 액션 타입별 카운트 계산
action_type_counts = data.groupby(['playerId', 'actionType']).size().unstack(fill_value=0).reset_index()

# 마지막 액션 시점 계산
last_action = data.groupby('playerId')['@timestamp'].max().reset_index()
last_action['recency'] = (end_time - last_action['@timestamp']).dt.days

# 특징 데이터 결합
features = action_counts.merge(action_type_counts, on='playerId')
features = features.merge(last_action[['playerId', 'recency']], on='playerId')

# 이탈 여부 레이블링 (예: 최근 7일 동안 활동이 없으면 이탈로 간주)
features['churn'] = features['recency'] > 7  # True/False 값을 1/0으로 변환
features['churn'] = features['churn'].astype(int)

# 클래스 확인 및 최소 두 개의 클래스가 존재하도록 필터링
if features['churn'].nunique() < 2:
    raise ValueError("데이터에 최소 두 개의 클래스(이탈/비이탈)가 필요합니다. 데이터에 이탈/비이탈 모두 포함되었는지 확인하세요.")

# 모델 입력 변수와 타겟 변수 정의
X = features.drop(columns=['playerId', 'churn'])
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

# 스케일러를 로컬에 저장
with open('/tmp/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 특징 목록 저장
feature_columns = X.columns.tolist()
with open('/tmp/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# MinIO에 업로드
bucket_name = 'models'
model_file_path = '/tmp/saved_model.pkl'
model_object_name = 'churn_model.pkl'
scaler_file_path = '/tmp/scaler.pkl'
scaler_object_name = 'scaler.pkl'

# 버킷이 존재하지 않으면 생성
try:
    s3.head_bucket(Bucket=bucket_name)
except Exception as e:
    s3.create_bucket(Bucket=bucket_name)

# 모델 파일 업로드
s3.upload_file(model_file_path, bucket_name, model_object_name)
# 스케일러 파일 업로드
s3.upload_file(scaler_file_path, bucket_name, scaler_object_name)
# 특징 목록 파일 업로드
s3.upload_file(feature_columns_file_path, bucket_name, feature_columns_object_name)

print("Model saved and uploaded to MinIO successfully.")