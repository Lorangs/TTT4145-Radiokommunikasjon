# Tuning Campaign

Started from `setup/config.yaml`

| Stage | Best candidate | Detect % | Header % | Payload % | Avg peak | Avg fine err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage0_baseline | baseline | 100.0 | 100.0 | 100.0 | 1.000 | 0.373 |
| stage1_gain | tx_0_rx_6 | 100.0 | 100.0 | 100.0 | 1.000 | 0.387 |
| stage2_preamble_cfo | current | 100.0 | 100.0 | 100.0 | 1.000 | 0.372 |
| stage3_gardner | faster_loop | 80.0 | 100.0 | 80.0 | 0.845 | 0.496 |
| stage4_carrier | phase_only_forced | 80.0 | 100.0 | 80.0 | 0.832 | 0.498 |
| stage5_equalizer | current | 100.0 | 100.0 | 100.0 | 1.000 | 0.369 |
| stage6_thresholds | stricter_thresholds | 100.0 | 100.0 | 100.0 | 1.000 | 0.368 |

Final config diff from baseline:

```yaml
receiver.rx_gain_dB: 6
gold_sequence.correlation_threshold: 0.75
gold_sequence.decode_candidate_min_peak: 0.7
synchronization.gardner_Kp: 0.004
synchronization.gardner_Ki: 2.0e-05
synchronization.header_carrier_apply_slope_enable: false
```
