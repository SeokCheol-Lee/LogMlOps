from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from elasticsearch7 import Elasticsearch
from datetime import datetime, timedelta
from pytz import timezone
from minio import Minio
import pickle
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Elasticsearch 설정
es = Elasticsearch(hosts=["192.168.49.2:32377"])

# 올바른 Minio 클라이언트 초기화
minio_endpoint = '192.168.49.2:31433'  
access_key = 'minioadmin'
secret_key = 'minioadmin'

# Minio 클라이언트 설정
minio_client = Minio(
    minio_endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=False  # HTTP 사용 시 secure=False
)

# MinIO에서 객체 로드 함수
def load_from_minio(object_name):
    try:
        logging.info(f"Loading {object_name} from MinIO...")
        bucket_name = 'models'

        # 객체 다운로드
        response = minio_client.get_object(bucket_name, object_name)
        body = response.read()
        response.close()
        response.release_conn()

        logging.info(f"Successfully loaded {object_name} from MinIO.")
        return pickle.loads(body)
    except Exception as e:
        logging.error(f"Error loading {object_name} from MinIO: {e}")
        raise

# 모델, 스케일러, 특징 목록 로드
logging.info("Loading model, scaler, and feature columns from MinIO...")
model = load_from_minio('churn_model.pkl')
scaler = load_from_minio('scaler.pkl')
feature_columns = load_from_minio('feature_columns.pkl')
logging.info("Successfully loaded model, scaler, and feature columns.")

@app.post("/predict/{playerId}")
def predict(playerId: int):
    logging.info(f"Received prediction request for playerId: {playerId}")
    try:
        # 1. Elasticsearch에서 playerId에 해당하는 데이터 로드
        player_id = str(playerId)  # Elasticsearch에서 playerId는 문자열로 저장될 수 있으므로 문자열로 변환
        logging.info(f"Querying Elasticsearch for playerId: {player_id}")

        # 최근 30일 동안의 데이터 가져오기
        utc_time = datetime.now(timezone('UTC'))
        utc_plus_9 = timezone('Asia/Seoul')
        end_time = utc_time.astimezone(utc_plus_9)
        start_time = end_time - timedelta(days=30)

        # 시간 범위를 ISO 포맷으로 변환
        start_time_iso = start_time.isoformat()
        end_time_iso = end_time.isoformat()

        # Elasticsearch 쿼리 정의 (특정 playerId에 대한 데이터만 가져옴)
        query_body = {
            "bool": {
                "must": [
                    {"term": {"playerId": player_id}},
                    {"range": {
                        "@timestamp": {
                            "gte": start_time_iso,
                            "lte": end_time_iso
                        }
                    }}
                ]
            }
        }

        # 데이터 검색 (query와 size 파라미터로 조정)
        res = es.search(index="logs", query=query_body, size=10000)
        hits = res['hits']['hits']
        logging.info(f"Elasticsearch query returned {len(hits)} hits for playerId: {player_id}")

        if len(hits) == 0:
            logging.warning(f"No data found for playerId: {player_id}")
            raise HTTPException(status_code=404, detail="Player data not found")

        # 2. 데이터 전처리
        data_list = []
        for hit in hits:
            source = hit['_source']
            data_list.append({
                'playerId': source['playerId'],
                'actionType': source['actionType'],
                '@timestamp': source['@timestamp']
            })

        data = pd.DataFrame(data_list)
        logging.info(f"Data for playerId {player_id}: {data.head()}")

        # timestamp를 datetime 형식으로 변환
        data['@timestamp'] = pd.to_datetime(data['@timestamp'])

        # 액션 카운트 계산
        action_count = data.shape[0]
        logging.info(f"Action count for playerId {player_id}: {action_count}")

        # 각 액션 타입별 카운트 계산
        action_type_counts = data.groupby('actionType').size().to_dict()
        logging.info(f"Action type counts for playerId {player_id}: {action_type_counts}")

        # 마지막 액션 시점 계산
        last_action_time = data['@timestamp'].max()
        recency = (end_time - last_action_time).days
        logging.info(f"Recency for playerId {player_id}: {recency} days")

        # 3. 예측을 위한 입력 데이터 준비
        input_data = {
            'action_count': action_count,
            'recency': recency
        }
        input_data.update(action_type_counts)

        # DataFrame으로 변환
        df = pd.DataFrame([input_data])
        logging.info(f"Prepared DataFrame for prediction: {df}")

        # 누락된 특징 추가 및 0으로 채움
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # 컬럼 순서 맞추기
        df = df[feature_columns]
        logging.info(f"DataFrame after matching feature columns: {df}")

        # 데이터 스케일링
        df_scaled = scaler.transform(df)
        logging.info(f"Scaled data: {df_scaled}")

        # 4. 예측 수행
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)[0][1]
        logging.info(f"Prediction: {prediction[0]}, Probability: {probability}")

        # 결과 반환
        return {'churn_prediction': int(prediction[0]), 'churn_probability': probability}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
