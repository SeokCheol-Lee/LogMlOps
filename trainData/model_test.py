import pandas as pd
import numpy as np
import pickle
import boto3
from botocore.client import Config
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random

# MinIO 설정
minio_endpoint = 'http://192.168.49.2:31433'
access_key = 'minioadmin'
secret_key = 'minioadmin'
bucket_name = 'models'

# MinIO 클라이언트 설정
s3 = boto3.client('s3',
                  endpoint_url=minio_endpoint,
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key,
                  config=Config(signature_version='s3v4'),
                  region_name='us-east-1')

# 파일 다운로드
model_file_path = '/tmp/churn_model.pkl'
scaler_file_path = '/tmp/scaler.pkl'
feature_columns_file_path = '/tmp/feature_columns.pkl'

s3.download_file(bucket_name, 'churn_model.pkl', model_file_path)
s3.download_file(bucket_name, 'scaler.pkl', scaler_file_path)
s3.download_file(bucket_name, 'feature_columns.pkl', feature_columns_file_path)

# 모델 및 스케일러 로드
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(feature_columns_file_path, 'rb') as columns_file:
    feature_columns = pickle.load(columns_file)

print("Model, Scaler, and Feature columns loaded successfully from MinIO.")

# Step 2: Generate synthetic test data (same structure as training data)
def generate_synthetic_data(num_records=100):
    player_ids = [f'player_{i}' for i in range(1, 101)]  # 100 players
    action_types = ['login', 'logout', 'purchase', 'view']  # Example action types
    data = []
    
    # Generate records for the past 30 days
    start_time = datetime.now() - timedelta(days=30)
    
    for _ in range(num_records):
        player_id = random.choice(player_ids)
        action_type = random.choice(action_types)
        timestamp = start_time + timedelta(days=random.uniform(0, 30))  # Random timestamp within 30 days
        data.append({
            'playerId': player_id,
            'actionType': action_type,
            '@timestamp': timestamp.isoformat()
        })
        
    return pd.DataFrame(data)

# Create test data
test_data = generate_synthetic_data()

# Preprocess the synthetic test data similarly as in the original code
test_data['@timestamp'] = pd.to_datetime(test_data['@timestamp'])

# Action counts
action_counts = test_data.groupby('playerId').size().reset_index(name='action_count')

# Action type counts
action_type_counts = test_data.groupby(['playerId', 'actionType']).size().unstack(fill_value=0).reset_index()

# Last action and recency calculation
last_action = test_data.groupby('playerId')['@timestamp'].max().reset_index()
end_time = datetime.now()
last_action['recency'] = (end_time - last_action['@timestamp']).dt.days

# Merge features
features_test = action_counts.merge(action_type_counts, on='playerId')
features_test = features_test.merge(last_action[['playerId', 'recency']], on='playerId')

# Ensure we have the same feature columns as during training, but keep 'playerId'
# Reindex without dropping 'playerId'
features_test_reindexed = features_test.reindex(columns=feature_columns + ['playerId'], fill_value=0)

# Step 3: Scale the test data and predict
# Drop 'playerId' only before scaling and predicting
X_test = features_test_reindexed.drop(columns=['playerId'])

# Scale the test data using the loaded scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions using the loaded model
y_pred = model.predict(X_test_scaled)

# Step 4: 결과 출력
features_test_reindexed['churn_prediction'] = y_pred
print("Churn predictions for test data:")
print(features_test_reindexed[['playerId', 'churn_prediction']])
