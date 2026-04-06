# stage5_equalizer

Short equalizer usefulness and acceptance sweep.

Carried config: `stage4_carrier:phase_only_forced`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 5 | 100.0 | 100.0 | 100.0 | 0.998 | 1.000 | 0.369 | 0.987 | 100.0 |
| eq_header_only | 5 | 60.0 | 100.0 | 60.0 | 0.655 | 0.697 | 0.610 | 0.651 | 0.0 |
| eq_5tap_highreg | 5 | 40.0 | 100.0 | 40.0 | 0.490 | 0.516 | 0.693 | 0.539 | 40.0 |
| eq_3tap_lowreg | 5 | 40.0 | 100.0 | 40.0 | 0.478 | 0.465 | 0.677 | 0.534 | 40.0 |
| equalizer_off | 5 | 40.0 | 100.0 | 40.0 | 0.479 | 0.413 | 0.713 | 0.478 | 0.0 |

Best candidate: `current`

Config diff from carried config:

```yaml
{}
```
