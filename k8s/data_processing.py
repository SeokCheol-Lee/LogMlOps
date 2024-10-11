import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

# Elasticsearch 연결 설정
es = Elasticsearch(hosts=["http://elasticsearch-service:9200"])

# 시간 범위 설정 (예: 최근 1일)
end_time = datetime.now()
start_time = end_time - timedelta(days=1)
start_time_iso = start_time.isoformat()
end_time_iso = end_time.isoformat()

# Elasticsearch 쿼리 설정
query = {
    "query": {
        "range": {
            "timestamp": {
                "gte": start_time_iso,
                "lte": end_time_iso
            }
        }
    },
    "size": 10000
}

index_name = "action_logs"

# 데이터 수집
res = es.search(index=index_name, body=query, scroll='2m')
scroll_id = res['_scroll_id']
hits = res['hits']['hits']

all_hits = []
all_hits.extend(hits)

while len(hits) > 0:
    res = es.scroll(scroll_id=scroll_id, scroll='2m')
    hits = res['hits']['hits']
    if not hits:
        break
    all_hits.extend(hits)

# DataFrame 변환
data_list = []
for hit in all_hits:
    source = hit['_source']
    data_list.append({
        'playerId': source['playerId'],
        'actionType': source['actionType'],
        'timestamp': source['timestamp']
    })

data = pd.DataFrame(data_list)

# 데이터 저장 (전처리된 데이터를 CSV로 저장)
data.to_csv('/tmp/processed_data.csv', index=False)
