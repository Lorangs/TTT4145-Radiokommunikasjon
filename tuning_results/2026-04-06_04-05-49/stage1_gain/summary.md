# stage1_gain

Compact local TX/RX gain sweep around the current 0 dB manual point.

Carried config: `stage0_baseline:baseline`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tx_0_rx_6 | 5 | 100.0 | 100.0 | 100.0 | 0.999 | 1.000 | 0.387 | 0.989 | 100.0 |
| current_0_0 | 5 | 100.0 | 100.0 | 80.0 | 0.993 | 1.000 | 0.424 | 0.989 | 100.0 |
| tx_m3_rx_6 | 5 | 80.0 | 100.0 | 80.0 | 0.814 | 0.832 | 0.462 | 0.814 | 80.0 |
| tx_m3_rx_0 | 5 | 80.0 | 100.0 | 80.0 | 0.905 | 0.819 | 0.435 | 0.922 | 80.0 |
| tx_m6_rx_6 | 5 | 40.0 | 100.0 | 40.0 | 0.480 | 0.523 | 0.703 | 0.503 | 40.0 |

Best candidate: `tx_0_rx_6`

Config diff from carried config:

```yaml
receiver.rx_gain_dB: 6
```
