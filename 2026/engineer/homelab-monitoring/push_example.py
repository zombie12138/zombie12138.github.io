"""
Push 模式示例：短命进程直接推送指标到 VictoriaMetrics。
适合定时任务、批处理脚本等跑完就退出的场景。

使用：
    pip install requests
    python push_example.py

不需要额外部署 Pushgateway，VictoriaMetrics 原生支持 Prometheus text import。
"""

import requests
import time

VM_URL = "http://victoriametrics:8428/api/v1/import/prometheus"
# Docker 网络外访问用: http://10.0.0.110:8428/api/v1/import/prometheus

# 模拟批处理任务
start = time.time()
record_count = 42  # 假设处理了 42 条记录
elapsed = time.time() - start

# 构造 Prometheus text 格式的指标
metrics = "\n".join(
    [
        f'batch_duration_seconds{{job="nightly_sync"}} {elapsed:.3f}',
        f'batch_records_total{{job="nightly_sync"}} {record_count}',
        f'batch_success{{job="nightly_sync"}} 1',
    ]
)

# 推送到 VictoriaMetrics
resp = requests.post(
    VM_URL,
    data=metrics,
    headers={"Content-Type": "text/plain"},
)
print(f"Push result: {resp.status_code}")

# 验证：
# curl 'http://10.0.0.110:8428/api/v1/query?query=batch_records_total'
