# TTT4145-Radiokommunikasjon

Vi velger frekvensbandet 863-870 MHz, siden dette er et "friomrade" i henhold
til europeisk EKOM. This frequency range also fits the ADALM-Pluto hardware
used in the project.

The current project focus is a low-rate Pluto SDR link for short IoT-style payloads.

## Active Link Setup

- Radio: ADALM-Pluto
- Carrier: `866.5e6`
- Modulation: BPSK
- Symbol rate: `125e3`
- Sample rate: `1e6`
- Samples per symbol: `8`
- RRC roll-off: `0.5`

## Active Burst Structure

The working over-the-air burst is intentionally simple:

- guard
- repeated timing preamble
- repeated Gold header
- coded payload
- guard


## Active TX Chain

- Datagram
- FEC
- Scrambler
- Interleaver
- Convolutional encoder
- BPSK symbol mapping
- prepend repeated timing preamble and repeated Gold header
- pulse shaping and transmit

## Active RX Chain

- capture from Pluto
- coarse FFT-based carrier correction
- matched filtering
- preamble/header alignment
- repeated-preamble CFO correction
- Gardner timing recovery
- fractional header timing refinement
- repeated-preamble and Gold-header symbol-slip correction
- header-based carrier phase correction
- short trained symbol-rate equalizer
- Costas loop fine correction
- Gold header removal and trim to the exact coded payload length
- demodulation and decoding back to Datagram

## Recommended Validation Path

`main.py` is currently not the reference validation path.

The scripts below are the ones the group should use when checking the current
system:

- `Hardware_tester.py`
- `frame_layout_check.py`
- `tuning_campaign.py`
- `pytest`

## Pluto Hardware Test

Use `Hardware_tester.py` for the main Pluto smoke test:

```bash
./.venv/bin/python Hardware_tester.py
```

Useful options:

```bash
./.venv/bin/python Hardware_tester.py --payload "hello pluto" --flush-buffers 3
./.venv/bin/python Hardware_tester.py --runs 20
./.venv/bin/python Hardware_tester.py --plots
```

The script will:

- verify the local FEC/scrambler/interleaver/convolutional round-trip
- connect to Pluto
- measure the RX noise floor and configure the synchronizer
- transmit one or more framed bursts and try to recover them over the air

For a loopback-style test, use either:

- a short RF cable with suitable attenuation between TX and RX
- or nearby antennas with low gain and enough separation to avoid overload

The hardware tester also saves decision-centric plots. These are usually more
useful than raw complex constellations when judging BPSK decode quality.

## Frame Layout Check

Use `frame_layout_check.py` to verify the current burst structure and header placement without needing a real Pluto capture:

```bash
./.venv/bin/python frame_layout_check.py
```

## Tuning Campaign

Use `tuning_campaign.py` only for controlled multi-run comparisons, not as the
everyday smoke-test path:

```bash
./.venv/bin/python tuning_campaign.py
```

## Tests

Run the local test suite with:

```bash
./.venv/bin/pytest -q
```
