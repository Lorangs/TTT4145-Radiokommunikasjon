# stage2_preamble_cfo

Preamble CFO and symbol-alignment sweep.

Carried config: `stage1_gain:tx_0_rx_6`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 5 | 100.0 | 100.0 | 100.0 | 0.998 | 1.000 | 0.372 | 0.987 | 100.0 |
| looser_scores | 5 | 60.0 | 100.0 | 60.0 | 0.721 | 0.703 | 0.567 | 0.783 | 80.0 |
| cfo_disabled | 5 | 60.0 | 100.0 | 60.0 | 0.671 | 0.665 | 0.595 | 0.000 | 60.0 |
| clamp_low | 5 | 40.0 | 100.0 | 40.0 | 0.503 | 0.535 | 0.627 | 0.553 | 80.0 |
| stricter_scores | 5 | 40.0 | 100.0 | 40.0 | 0.566 | 0.452 | 0.604 | 0.606 | 40.0 |

Best candidate: `current`

Config diff from carried config:

```yaml
{}
```
