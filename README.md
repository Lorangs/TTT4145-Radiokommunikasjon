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
-- MODULATOR
-- PILOT / GOLD
-- TX
-- RX
-- COARSE SYMBOL SYNC
-- TIMING SYNC
-- FINE SYMBOL SYNC
-- FRAME SYNC
-- REMOVE GOLD / PILOT
-- DEMODULATOR
-- CONVOLUTIONAL DECODER
-- DEINTERLEAVER
-- DESCRAMBLER
-- FEC
-- Datagram

