# stage6_thresholds

Detection and decode-threshold robustness sweep.

Carried config: `stage5_equalizer:current`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stricter_thresholds | 5 | 100.0 | 100.0 | 100.0 | 0.996 | 1.000 | 0.368 | 0.987 | 100.0 |
| lower_decode_gate | 5 | 80.0 | 100.0 | 80.0 | 0.838 | 0.845 | 0.405 | 0.836 | 80.0 |
| lower_corr_decode | 5 | 60.0 | 100.0 | 60.0 | 0.660 | 0.677 | 0.617 | 0.631 | 80.0 |
| softer_alignment | 5 | 40.0 | 100.0 | 40.0 | 0.477 | 0.465 | 0.529 | 0.490 | 40.0 |
| current | 5 | 40.0 | 100.0 | 40.0 | 0.494 | 0.458 | 0.635 | 0.530 | 60.0 |

Best candidate: `stricter_thresholds`

Config diff from carried config:

```yaml
gold_sequence.correlation_threshold: 0.75
gold_sequence.decode_candidate_min_peak: 0.7
```
