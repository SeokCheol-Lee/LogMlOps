from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

spark = SparkSession.builder.appName("LogProcessing").getOrCreate()

# Elasticsearch에서 로그 데이터 로드
logs_df = spark.read \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch-service") \
    .option("es.port", "9200") \
    .load("filebeat-*")  # 인덱스와 타입을 정확히 지정

# 로그 패턴 정의
log_pattern = r'PlayerID: (\d+), ActionType: (\w+), Timestamp: (\d+)'

# 로그 데이터 파싱
parsed_logs = logs_df.select(
    regexp_extract('message', log_pattern, 1).alias('PlayerID'),
    regexp_extract('message', log_pattern, 2).alias('ActionType'),
    regexp_extract('message', log_pattern, 3).alias('Timestamp')
)

# 데이터 타입 변환
parsed_logs = parsed_logs.withColumn("PlayerID", parsed_logs["PlayerID"].cast("integer")) \
                         .withColumn("Timestamp", parsed_logs["Timestamp"].cast("long"))

# 필요한 특징 추출 (예: 플레이어별 액션 카운트)
feature_df = parsed_logs.groupBy('PlayerID', 'ActionType').count()

# 결과 저장 (예: HDFS 또는 S3)
feature_df.write.format("parquet").save("/path/to/features")

spark.stop()
