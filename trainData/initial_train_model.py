import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from minio import Minio

# 1. MinIO 서버에 연결
minioClient = Minio(
    '192.168.49.2:30000',  # MinIO 서버의 IP와 포트
    access_key='minioadmin',  # MinIO 액세스 키
    secret_key='minioadmin',  # MinIO 비밀 키
    region='ap-northeast-2',  # MinIO의 리전 설정
    secure=False  # HTTPS를 사용하지 않음 (로컬 환경)
)

# 2. 로그 데이터를 랜덤하게 생성하는 함수 (최초 학습용)
def generate_random_logs(num_players, num_logs_per_player):
    logs = []
    actions = ["MOVE", "JUMP", "SHOOT", "RUN"]  # 행동 타입 목록
    for player_id in range(1, num_players + 1):
        for _ in range(num_logs_per_player):
            action = random.choice(actions)  # 무작위로 행동을 선택
            timestamp = pd.Timestamp.now().timestamp()  # 현재 타임스탬프 생성
            logs.append({
                "playerId": player_id,  # 플레이어 ID
                "actionType": action,  # 행동 타입
                "timestamp": int(timestamp)  # 타임스탬프 (정수형 변환)
            })
    return logs

# 3. 로그 데이터를 생성하고 DataFrame으로 변환
random_logs = generate_random_logs(num_players=100, num_logs_per_player=20)
log_df = pd.DataFrame(random_logs)

# 4. playerId별 로그 횟수를 계산하여 학습 데이터 생성
player_log_counts = log_df.groupby("playerId").size().reset_index(name="log_count")

# 로그 횟수에 기반한 레벨 생성 (랜덤성을 포함)
player_levels = player_log_counts.copy()
player_levels["level"] = player_log_counts["log_count"].apply(
    lambda x: min(100, max(1, int(x + random.gauss(0, 5))))  # Gaussian 노이즈 추가
)

# 5. 모델 학습을 위한 데이터 분리
X = player_levels[["log_count"]]  # feature: 로그 횟수
y = player_levels["level"]  # target: 레벨

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Linear Regression 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 모델 평가 (MSE 계산)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Initial Model MSE: {mse:.2f}")  # 최초 학습 모델의 MSE 출력

# 8. MinIO에 버킷이 없으면 생성
bucket_name = "models"
found = minioClient.bucket_exists(bucket_name)
if not found:
    minioClient.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created successfully.")  # 버킷 생성 메시지
else:
    print(f"Bucket '{bucket_name}' already exists.")  # 버킷이 이미 존재하는 경우

# 9. 학습된 모델을 로컬에 저장
local_model_dir = os.path.expanduser("~/models")  # 홈 디렉토리 내에 모델 디렉토리 설정
local_model_path = os.path.join(local_model_dir, "trained_model_initial.pkl")  # 모델 파일 경로

# 경로가 없으면 디렉토리 생성
if not os.path.exists(local_model_dir):
    os.makedirs(local_model_dir)
    print(f"Directory '{local_model_dir}' created successfully.")  # 디렉토리 생성 메시지

# 학습된 모델을 로컬에 저장
joblib.dump(model, local_model_path)

# 10. 저장된 모델을 MinIO에 업로드
minioClient.fput_object(bucket_name, "trained_model_initial.pkl", local_model_path)
print("Model uploaded to MinIO successfully.")  # 모델 업로드 성공 메시지
