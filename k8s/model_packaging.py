import boto3
from botocore.client import Config
import tarfile

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

# 모델을 tar.gz로 압축
model_tar_path = '/tmp/saved_model.tar.gz'
with tarfile.open(model_tar_path, "w:gz") as tar:
    tar.add('/tmp/saved_model', arcname='saved_model')

# MinIO에 업로드
bucket_name = 'model-bucket'
s3.upload_file(model_tar_path, bucket_name, 'saved_model.tar.gz')
