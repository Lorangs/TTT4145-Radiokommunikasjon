# TTT4145-Radiokommunikasjon

Vi velger frekvensbånd 863-870MHz, da dette er tilegnet "Fri Område" i henhold til europeisk EKOM.
This frequenzy range fits the ADALM PLUTO antenna specs perfectly. 

The dataset / type we will communicate will consist low datarate Iot datasets of 8-32 bit frames.


# From Chatgpt breakdown of the repo structure and functionality:
This repo is a Python SDR chat prototype for ADALM-Pluto in the 863-870 MHz ISM band.
It sends short text payloads as framed datagrams over BPSK/QPSK with Barker preamble, RRC filtering, and basic ACK handling.

Core Runtime

Entry point: main.py
Main app class: SDRChatApp in main.py
Threads:
RX thread: receive IQ, filter, synchronize, detect Barker, demodulate, enqueue ACK/data
TX thread: dequeue datagram, modulate, add Barker, upsample/filter, transmit
TUI thread: terminal chat UI + user input (/quit)
Signal/Data Pipeline

User text -> Datagram (datagram.py)
Datagram bytes -> symbols (modulation.py)
Add preamble (barker_detection.py)
Upsample + pulse shaping (modulation.py, filter.py)
Send via Pluto (sdr_transciever.py)
RX does inverse chain + timing recovery (synchronize.py) and preamble detection
Important Modules

main.py: orchestration, threading, logging, lifecycle
datagram.py: message format (msgType.DATA/ACK), CRC16, pack/unpack
modulation.py: BPSK/QPSK mapping + symbol up/downsample
filter.py: RRC generation, software filter, optional hardware filter file generation
synchronize.py: coarse timing sync (Mueller & Muller style)
barker_detection.py: preamble insert/detect/remove
sdr_transciever.py: pyadi-iio Pluto config + TX/RX I/O
chat_tui.py: simple terminal UI
Config + Dependencies

Runtime config: config.yaml
Python deps: requirements.txt
Debug mode in config enables live plotting (sdr_plots.py, PyQt/Matplotlib)
Auxiliary/Experimental

simple_tester.py: end-to-end smoke test script
rrc_filter_generator.py: older filter utility
tx_spectrum_analysis/ and rx_spectrum_analysis/: spectrum analysis scripts/data
How a new dev should start

Create venv and install requirements.txt
Set Pluto IP + radio parameters in config.yaml
Start with python main.py
If debugging signal quality, keep radio.debug_mode: True and use plots/logs in log/
Current caveats worth knowing

Naming typo is consistent: sdr_transciever.py (not transceiver)
Some auxiliary scripts look stale/incomplete versus current config keys (e.g. tx_test.py, rx_measure_spectrum.py)
README.md is minimal; main.py + config.yaml are the real source of truth.