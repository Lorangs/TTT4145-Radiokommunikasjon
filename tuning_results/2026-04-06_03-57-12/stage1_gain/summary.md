# stage1_gain

Compact local TX/RX gain sweep around the current 0 dB manual point.

Carried config: `stage0_baseline:baseline`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tx_m6_rx_6 | 5 | 100.0 | 100.0 | 100.0 | 0.999 | 1.000 | 0.371 | 0.989 | 100.0 |
| current_0_0 | 5 | 100.0 | 100.0 | 100.0 | 0.956 | 1.000 | 0.375 | 0.989 | 100.0 |
| tx_0_rx_6 | 5 | 100.0 | 100.0 | 100.0 | 0.998 | 1.000 | 0.385 | 0.985 | 100.0 |
| tx_m3_rx_6 | 5 | 80.0 | 100.0 | 80.0 | 0.860 | 0.845 | 0.417 | 0.852 | 80.0 |
| tx_m3_rx_0 | 5 | 80.0 | 100.0 | 80.0 | 0.863 | 0.839 | 0.426 | 0.851 | 80.0 |

Best candidate: `tx_m6_rx_6`

Config diff from carried config:

```yaml
transmitter.tx_gain_dB: -6
receiver.rx_gain_dB: 6
```
