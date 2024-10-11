import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import boto3
from botocore.client import Config
import tarfile
from pytz import timezone

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
res = es.search(index=index_name, query=query_body, scroll='2m', size=10000)
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

# 모델 입력 변수와 타겟 변수 정의
X = features.drop(columns=['playerId', 'churn'])
y = features['churn']

# 3. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 7. 모델 저장 (MinIO에 저장)
# MinIO 설정
minio_endpoint = '192.168.49.2:31433'
access_key = 'minioadmin'  
secret_key = 'minioadmin' 

s3 = boto3.client('s3',
                  endpoint_url=minio_endpoint,
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key,
                  config=Config(signature_version='s3v4'),
                  region_name='us-east-1')

# 모델 로컬에 저장
model.save('/tmp/saved_model')
