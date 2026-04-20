# TTT4145-Radiokommunikasjon

ADALM-Pluto-based digital radio link for short packet transmission in the 863-870 MHz band.

The project contains a configurable TX/RX chain, framing with Gold-code headers, forward error correction, synchronization, and an optional live plotting/debug workflow.

## Overview

The application is driven by [main.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/main.py) and configured through [setup/config.yaml](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/setup/config.yaml).

Current default behavior in the repo:

- Radio: ADALM-Pluto via `pyadi-iio`
- Carrier: `866.5e6`
- Modulation: `QPSK`
- Symbol rate: `125e3`
- Sample rate: `1e6`
- Samples per symbol: `8`
- Pulse shaping: root-raised-cosine
- Link layer: optional ACK and retransmit support

## Signal Chain

Transmit path:

- Datagram packing
- Reed-Solomon coding
- Interleaving
- Scrambling
- Convolutional coding
- Symbol mapping
- Gold-code framing
- Upsampling and pulse shaping
- SDR transmission

Receive path:

- Pluto SDR capture
- Coarse frequency correction
- Matched filtering
- Gardner timing recovery
- Fine frequency correction with Costas loop
- Gold-code detection and rotation handling
- Optional short equalizer trained on the known header
- Demodulation
- Convolutional, descrambler, deinterleaver, and Reed-Solomon decode
- Datagram unpacking

## Repository Layout

- [main.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/main.py): application entry point, thread orchestration, TX/RX loops
- [setup/config.yaml](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/setup/config.yaml): runtime configuration
- [sdr_transciever.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/sdr_transciever.py): Pluto SDR setup and hardware I/O
- [modulation.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/modulation.py): symbol mapping and upsampling
- [filter.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/filter.py): RRC filter generation and application
- [synchronize.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/synchronize.py): coarse/fine frequency and timing synchronization
- [gold_detection.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/gold_detection.py): Gold header insertion, detection, and rotation estimation
- [equalizer.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/equalizer.py): short trained complex FIR equalizer
- [datagram.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/datagram.py): packet format and serialization
- [chat_tui.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/chat_tui.py): terminal chat interface
- [sdr_plots.py](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/sdr_plots.py): live and static debug plots

## Requirements

- Python 3.11 or newer recommended
- ADALM-Pluto reachable from the machine running the app
- Pluto configured with an IP address that matches `radio.ip_address` in the config

Python dependencies are listed in [requirements.txt](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/requirements.txt), including:

- `pyadi-iio`
- `numpy`
- `scipy`
- `numba`
- `reedsolo`
- `scikit-commpy`
- `matplotlib`
- `PyQt6`
- `pyqtgraph`

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Update the Pluto address and radio parameters in [setup/config.yaml](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/setup/config.yaml) before running.

## Running

Start the application with:

```bash
./.venv/bin/python main.py
```

When the app starts successfully, it:

- loads configuration
- initializes the SDR, modulation, coding, synchronization, and plotting modules
- measures the RX noise floor
- starts RX, TX, TUI, and ACK-timeout worker threads

In terminal mode, type a message and press Enter to queue it for transmission.

Available command:

- `/quit` to stop the application

## Configuration

Most tuning happens in [setup/config.yaml](/Users/bendiknygard/Documents/GitHub/TTT4145-Radiokommunikasjon/setup/config.yaml).

Important sections:

- `radio`: Pluto address, queue size, debug plotting enable
- `link_control`: ACK, NACK, retransmit, and pending-packet tracking
- `transmitter`: TX gain, carrier, bandwidth, guard length, burst scaling
- `receiver`: RX gain, AGC/manual mode, carrier, bandwidth, buffer size
- `modulation`: modulation type, order, symbol rate, samples per symbol, sample rate
- `coding`: Reed-Solomon, convolutional code, and scrambler settings
- `filter`: RRC settings and optional hardware filter file
- `gold_sequence`: Gold code length, index, and correlation threshold
- `synchronization`: FFT size, detection threshold, Costas and Gardner loop settings, equalizer enable
- `plotter` and `plot_capture`: live plot behavior and optional saved debug figures

## Debugging And Plots

Set `radio.debug_mode: true` to enable the Qt-based plot windows.

The plotting layer can show:

- time-domain IQ
- frequency-domain PSD
- waterfall view
- constellation plots
- saved static debug plots for selected RX and TX events

Plot capture output is written to the directory configured in `plot_capture.output_dir`.

## Notes For Over-The-Air Testing

- TX and RX carriers should normally match unless you are intentionally testing offset behavior.
- The frequency labels in plots come from the plotting configuration, not directly from the SDR LO values.
- Two-radio testing is more sensitive to gain, bandwidth, timing, and carrier offset than single-radio or close-coupled tests.
- If the air link is unstable, start by checking gain mode, RX bandwidth, modulation choice, and Gold correlation threshold.

## Known Limits

- The current application is centered on `main.py`; older validation scripts referenced in earlier project notes are not part of this repository snapshot.
- Debug plotting depends on a working Qt environment.
- Pluto hardware filtering support exists in the config, but the main software path still assumes software filtering by default.
