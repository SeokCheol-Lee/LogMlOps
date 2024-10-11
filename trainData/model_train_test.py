import pandas as pd
import numpy as np
import pickle
import boto3
from botocore.client import Config
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta
import random
import os

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

# Step 1: Generate synthetic data
def generate_synthetic_data(num_records=1000):
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

# Step 2: Preprocess the data and train the model
data = generate_synthetic_data()

# Preprocess DataFrame as in the original code
data['@timestamp'] = pd.to_datetime(data['@timestamp'])

# Action counts
action_counts = data.groupby('playerId').size().reset_index(name='action_count')

# Action type counts
action_type_counts = data.groupby(['playerId', 'actionType']).size().unstack(fill_value=0).reset_index()

# Last action and recency calculation
last_action = data.groupby('playerId')['@timestamp'].max().reset_index()
end_time = datetime.now()
last_action['recency'] = (end_time - last_action['@timestamp']).dt.days

# Merge features
features = action_counts.merge(action_type_counts, on='playerId')
features = features.merge(last_action[['playerId', 'recency']], on='playerId')

# Churn label: Players who haven't acted in 7+ days are marked as churned
features['churn'] = features['recency'] > 7
features['churn'] = features['churn'].astype(int)

# Define model input variables and target variable
X = features.drop(columns=['playerId', 'churn'])
y = features['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Step 5: Save the model, scaler, and feature columns locally
model_file_path = '/tmp/churn_model.pkl'
scaler_file_path = '/tmp/scaler.pkl'
feature_columns_file_path = '/tmp/feature_columns.pkl'

with open(model_file_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(scaler_file_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

feature_columns = X.columns.tolist()
with open(feature_columns_file_path, 'wb') as feature_columns_file:
    pickle.dump(feature_columns, feature_columns_file)

print("Model, scaler, and feature columns saved locally.")

# Step 6: Upload the model to MinIO
try:
    s3.head_bucket(Bucket=bucket_name)
except Exception as e:
    s3.create_bucket(Bucket=bucket_name)

s3.upload_file(model_file_path, bucket_name, 'churn_model.pkl')
s3.upload_file(scaler_file_path, bucket_name, 'scaler.pkl')
s3.upload_file(feature_columns_file_path, bucket_name, 'feature_columns.pkl')

print("Model, scaler, and feature columns uploaded to MinIO.")
