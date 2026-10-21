# 压测报告 mock_gateway_hot30（target=gateway, stream=False, max_tokens=256）

| concurrency | QPS | P50_ms | P95_ms | P99_ms | TTFT_P50_ms | out_tok_per_s | error_rate | http_429 | gpu_peak_mem_MB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.56 | 715.4 | 1279.0 | 1533.0 | None | 96.4 | 0.0 | 0 | None |
| 4 | 7.68 | 531.7 | 1294.6 | 1442.1 | None | 441.1 | 0.0 | 0 | None |
| 8 | 13.23 | 629.1 | 1405.4 | 1677.4 | None | 763.6 | 0.0 | 0 | None |
| 16 | 15.88 | 967.2 | 1616.9 | 1694.7 | None | 1035.3 | 0.0 | 0 | None |
| 32 | 22.2 | 1104.8 | 1970.6 | 1988.0 | None | 1346.2 | 0.0 | 0 | None |
