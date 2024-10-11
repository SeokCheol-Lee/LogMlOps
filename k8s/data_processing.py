import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
from pytz import timezone

# Elasticsearch 연결 설정
es = Elasticsearch(hosts=["192.168.49.2:32377"])

# 시간 범위 설정 (예: 최근 1일)
utc_time = datetime.now(timezone('UTC'))
utc_plus_9 = timezone('Asia/Seoul')
end_time = utc_time.astimezone(utc_plus_9)
start_time = end_time - timedelta(days=1)

# Elasticsearch 쿼리 정의
query_body = {
    "range": {
        "@timestamp": {
            "gte": start_time,
            "lte": end_time
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

# DataFrame 변환
data_list = []
for hit in all_hits:
    source = hit['_source']
    data_list.append({
        'playerId': source['playerId'],
        'actionType': source['actionType'],
        '@timestamp': source['@timestamp']
    })

data = pd.DataFrame(data_list)

# 데이터 저장 (전처리된 데이터를 CSV로 저장)
data.to_csv('/tmp/processed_data.csv', index=False)
