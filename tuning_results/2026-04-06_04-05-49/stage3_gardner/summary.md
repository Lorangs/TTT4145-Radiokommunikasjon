# stage3_gardner

Gardner timing-loop sweep.

Carried config: `stage2_preamble_cfo:current`

| Candidate | Runs | Detect % | Header % | Payload % | Avg header peak | Avg final peak | Avg fine err | Avg CFO score | Eq accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| faster_loop | 5 | 80.0 | 100.0 | 80.0 | 0.818 | 0.845 | 0.496 | 0.826 | 80.0 |
| looser_gate_long | 5 | 60.0 | 100.0 | 60.0 | 0.670 | 0.697 | 0.570 | 0.758 | 80.0 |
| slower_loop | 5 | 60.0 | 100.0 | 60.0 | 0.655 | 0.677 | 0.623 | 0.655 | 60.0 |
| current | 5 | 40.0 | 100.0 | 40.0 | 0.501 | 0.529 | 0.756 | 0.495 | 40.0 |
| tighter_gate_short | 5 | 40.0 | 100.0 | 40.0 | 0.489 | 0.516 | 0.692 | 0.464 | 40.0 |

Best candidate: `faster_loop`

Config diff from carried config:

```yaml
synchronization.gardner_Kp: 0.004
synchronization.gardner_Ki: 2.0e-05
```
