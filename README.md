# TTT4145-Radiokommunikasjon

Vi velger frekvensbånd 863-870MHz, da dette er tilegnet "Fri Område" i henhold til europeisk EKOM.
This frequenzy range fits the ADALM PLUTO antenna specs perfectly. 

The dataset / type we will communicate will consist low datarate Iot datasets of 8-32 bit frames.


----- PIPELINE ----- 

-- Datagram
-- FEC
-- SCRAMBLER
-- INTERLEAVER
-- CONVOLUTIONAL ENCODER
-- MODULATOR # QPSK
-- PILOT / GOLD # BPSK, added as symbols in front and back of the frame, used for synchronization and channel estimation
-- TX
-- RX
-- COARSE SYMBOL SYNC
-- TIMING SYNC
-- FINE SYMBOL SYNC
-- FRAME SYNC
-- REMOVE GOLD / PILOT # BPSK
-- DEMODULATOR
-- CONVOLUTIONAL DECODER
-- DEINTERLEAVER
-- DESCRAMBLER
-- FEC
-- Datagram

----- PLUTO HARDWARE TEST -----

Use `Hardware_tester.py` to run a local Pluto smoke test:

1. Connect the Pluto and make sure the IP in `setup/config.yaml` matches your device.
2. Use either:
`./.venv/bin/python Hardware_tester.py`

Optional flags:
`./.venv/bin/python Hardware_tester.py --payload "hello pluto" --rx-buffer-size 65536 --flush-buffers 3`

For a loopback-style test, use either:
- a short RF cable with suitable attenuation between TX and RX
- or nearby antennas with low gain and enough separation to avoid overload

The script will:
- verify the local FEC/scrambler/interleaver/convolutional pipeline
- connect to Pluto
- measure the RX noise floor and configure the synchronizer
- transmit one framed burst and try to recover it over the air

Use plots only when needed:
`./.venv/bin/python Hardware_tester.py --plots`
