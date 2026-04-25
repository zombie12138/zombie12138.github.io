"""
Pull 模式示例：服务暴露 /metrics 端点，vmagent 定时 scrape。
适合长期运行的服务。

使用：
    pip install prometheus_client
    python pull_example.py

    # 验证
    curl http://localhost:9527/metrics

接入 vmagent（monitoring/vmagent/scrape.yml）：
    - job_name: myapp
      static_configs:
        - targets: ["myapp-host:9527"]
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time, random

# 定义指标
requests_total = Counter(
    "myapp_requests_total",
    "Total requests processed",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "myapp_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)

queue_depth = Gauge("myapp_queue_depth", "Tasks waiting in queue")

# 暴露 /metrics 端点
start_http_server(9527)
print("Metrics server started on :9527/metrics")

# 模拟业务
while True:
    status = random.choice(["200"] * 9 + ["500"])
    requests_total.labels(method="GET", endpoint="/api", status=status).inc()

    with request_duration.time():
        time.sleep(random.uniform(0.01, 0.5))

    queue_depth.set(random.randint(0, 20))
