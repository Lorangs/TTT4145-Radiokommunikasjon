# stage4_carrier

Costas and header-carrier sweep.

Carried config: `stage3_gardner:faster_loop`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase_only_forced | 5 | 80.0 | 100.0 | 80.0 | 0.821 | 0.832 | 0.498 | 0.814 | 80.0 |
| conservative_slope | 5 | 80.0 | 100.0 | 80.0 | 0.825 | 0.800 | 0.495 | 0.817 | 80.0 |
| faster_costas | 5 | 60.0 | 100.0 | 60.0 | 0.665 | 0.645 | 0.571 | 0.676 | 80.0 |
| current | 5 | 20.0 | 100.0 | 20.0 | 0.330 | 0.381 | 0.870 | 0.356 | 60.0 |
| slower_costas | 5 | 20.0 | 100.0 | 20.0 | 0.306 | 0.329 | 0.787 | 0.381 | 20.0 |

Best candidate: `phase_only_forced`

Config diff from carried config:

```yaml
synchronization.header_carrier_apply_slope_enable: false
```
