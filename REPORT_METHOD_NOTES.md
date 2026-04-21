# Method Notes For Report Writing

This document summarizes the algorithms that are actually used in the current repository so the method section can describe the implemented system, not just the intended design.

Relevant source files:

- `main.py`
- `datagram.py`
- `forward_error_correction.py`
- `interleaver.py`
- `scrambler.py`
- `convolutional_coder.py`
- `modulation.py`
- `filter.py`
- `gold_code.py`
- `gold_detection.py`
- `synchronize.py`
- `equalizer.py`
- `sdr_transciever.py`

## 1. End-to-end communication chain

Transmit chain:

1. Application data is packed into a fixed-size datagram.
2. A CRC-16 is computed over the logical datagram fields.
3. Reed-Solomon block coding adds symbol-level redundancy.
4. A deterministic bit interleaver permutes the coded bitstream.
5. A synchronous additive LFSR scrambler whitens the bitstream.
6. A convolutional encoder adds additional redundancy.
7. The coded bits are mapped to BPSK or QPSK symbols.
8. A Gold-code header is added before and after the payload.
9. The symbol stream is upsampled and pulse-shaped by a root-raised-cosine filter.
10. Guard samples are inserted and the burst is peak-normalized before Pluto transmission.

Receive chain:

1. Pluto captures complex baseband samples.
2. A coarse carrier-frequency-offset estimator uses an FFT on the m-th power of the signal.
3. The signal is matched-filtered with the same RRC pulse shape.
4. Symbol timing is recovered with a Gardner timing-recovery loop using cubic interpolation.
5. Residual carrier/phase error is corrected by a Costas loop.
6. A Gold-code detector performs normalized correlation to locate the frame.
7. The Gold header is also used to estimate the remaining constellation rotation.
8. An optional 3-tap complex equalizer is trained from the known Gold header.
9. The payload symbols are demodulated with hard decisions.
10. The inverse coding chain is applied: Viterbi decode, descramble, deinterleave, Reed-Solomon decode.
11. The datagram is unpacked and verified with CRC-16.

## 2. Packet format and link layer

### Fixed-length datagram

The packet format is defined in `datagram.py` and contains:

- `msg_id` (1 byte)
- `msg_type` (1 byte)
- `timestamp_ms` (4 bytes)
- `payload_length` (1 byte)
- `payload_crc16` (2 bytes)
- `payload` (23 bytes maximum, zero-padded on the air interface)

The runtime therefore sends a fixed 32-byte datagram on every transmission.

### CRC-16 integrity check

The payload integrity check uses CRC-16-CCITT (`crc_hqx`) with initial value `0xFFFF`. The CRC is computed over:

- message ID
- message type
- timestamp
- logical payload length
- logical payload bytes

This gives an end-to-end checksum on the logical datagram contents after all channel decoding is finished.

### ACK / retransmission logic

The link layer in `main.py` includes optional ACK-based reliability:

- DATA packets can be tracked until an ACK is received.
- If an ACK is not received within a timeout, the packet is retransmitted.
- The implementation retransmits the oldest pending packet first.
- Maximum retry count is configurable.

This is a simple ARQ mechanism on top of the physical layer.

## 3. Forward error correction and bit-domain processing

### Reed-Solomon block code

`forward_error_correction.py` uses the `reedsolo` library:

- `RSCodec(rs_num_ecc)` is applied to the full datagram byte vector.
- With the current configuration, `rs_num_ecc = 8`, so 8 parity bytes are added.
- A 32-byte datagram therefore becomes 40 bytes before the next stage.

This is the outer code in the concatenated coding chain.

Suggested report wording:

"An outer Reed-Solomon code was applied at the byte level to correct burst-like symbol errors remaining after demodulation and inner decoding."

### Bit interleaver

`interleaver.py` implements a deterministic bit interleaver:

- packed bytes are unpacked to bits in little-endian order
- a pseudo-random permutation is generated from a fixed seed
- the permutation depends on the current frame length
- the receiver reconstructs the same permutation and applies the inverse mapping

Purpose:

- spread clustered channel errors across the frame
- make the outer Reed-Solomon decoder see a more distributed error pattern

Implementation note:

- `setup/config.yaml` does not currently define `interleaver_seed`, so the code uses the default seed `42`

### LFSR scrambler

`scrambler.py` implements a synchronous additive scrambler:

- a linear-feedback shift register generates a pseudo-random byte sequence
- each payload byte is XORed with that sequence
- the same operation at the receiver descrambles the data

The scrambler is seeded once per packet from the configured non-zero seed.

Purpose:

- reduce long runs of identical bits
- make the transmitted spectrum flatter
- help downstream synchronization and detection behave more consistently

### Convolutional code and Viterbi decoder

`convolutional_coder.py` implements the inner code:

- configurable constraint length `K`
- configurable code rate such as `1/2`, `1/3`, or `1/4`
- generator polynomials are given in octal form
- the current configuration uses `K = 7` and rate `1/3`

Encoding:

- input bytes are unpacked to bits
- the encoder shifts one input bit at a time through the register
- `n` parity bits are generated per input bit
- `K-1` zero tail bits are appended to terminate the trellis

Decoding:

- hard-decision Viterbi decoding is used
- branch metrics are Hamming distances between expected and received coded bits
- traceback starts from the all-zero final state because the encoder is terminated with zero tail bits

Suggested report wording:

"An inner terminated convolutional code was decoded by a hard-decision Viterbi algorithm, forming a concatenated FEC structure together with the outer Reed-Solomon code."

## 4. Modulation and pulse shaping

### BPSK / QPSK symbol mapping

`modulation.py` supports BPSK and QPSK in the custom synchronization/framing path.

BPSK mapping:

- bit `0 -> +1`
- bit `1 -> -1`

QPSK mapping:

- two bits are mapped to the sign of the in-phase and quadrature branches
- the implemented constellation points are `+/-1 +/- j`
- this means the QPSK constellation is not normalized by `1/sqrt(2)`; the raw symbol energy is therefore 2

Demodulation is hard-decision slicing on the sign of the real and imaginary components.

### Upsampling

The symbol stream is upsampled by zero insertion:

- one symbol every `samples_per_symbol`
- zeros in the intermediate samples

With the current configuration:

- symbol rate = `125 ksymbols/s`
- samples per symbol = `8`
- sample rate = `1 MS/s`

### Root-raised-cosine pulse shaping

`filter.py` generates a root-raised-cosine (RRC) pulse:

- roll-off factor `alpha = 0.5`
- filter span `10` symbols
- `8` samples per symbol
- coefficients are normalized to unit energy

The transmitter applies the RRC filter for pulse shaping and the receiver applies the same filter as a matched filter. Together, the TX and RX filters form an overall raised-cosine response intended to reduce inter-symbol interference.

## 5. Frame synchronization and preamble design

### Gold code generation

`gold_code.py` generates Gold-code families from preferred pairs of m-sequences:

- supported lengths are `2^m - 1`
- the current configuration uses code length `31`
- the code family is generated from two LFSR m-sequences and their shifted XOR combinations

### Gold framing

`gold_detection.py` adds the selected Gold sequence:

- once before the payload
- once after the payload

Frame format at symbol level:

`[leading Gold][payload][trailing Gold]`

This gives both a robust frame marker and a known training sequence for phase estimation and equalization.

### Gold detection by normalized correlation

At the receiver, frame detection is performed by normalized sliding correlation:

`C[k] = |sum_n r[k+n] g*[n]| / sqrt(E_g sum_n |r[k+n]|^2)`

where:

- `r[k]` is the received symbol stream
- `g[n]` is the known Gold sequence
- `E_g` is the Gold-sequence energy

The implementation:

- computes correlation against all allowed phase-rotated Gold templates
- keeps only peaks above a configurable threshold
- checks that the candidate index can fit a full frame
- selects the earliest valid candidate to avoid locking to the trailing Gold sequence

### Rotation estimation from the Gold header

After detection, the receiver estimates residual phase rotation from the leading Gold sequence:

- it correlates the received header with the unrotated Gold reference
- it extracts the correlation phase
- it snaps the correction to the nearest allowed constellation rotation

For QPSK, the allowed rotations are:

- `0°`
- `90°`
- `180°`
- `270°`

This gives a coarse decision-directed phase ambiguity resolver before payload demodulation.

Implementation note:

- for QPSK, the Gold symbols are formed by placing the same BPSK Gold sequence on both the I and Q branches, so the header lies on the diagonal QPSK points

## 6. Synchronization algorithms

### Noise-floor estimation and burst gating

Before runtime reception starts, `sdr_transciever.py` measures the noise floor:

- several RX buffers are captured
- average sample power is computed for each
- the mean power is converted to dB

During coarse synchronization, a burst is only processed further if the estimated signal power is above:

`noise_floor_dB + signal_power_threshold_dB`

This provides a simple signal-presence detector.

### Coarse carrier-frequency synchronization

`synchronize.py` uses an FFT-based m-th power estimator:

- the received signal is raised to the modulation order
- BPSK uses second power
- QPSK uses fourth power
- the peak of the FFT magnitude estimates the carrier offset
- the estimated offset is divided by the modulation order and removed from the signal

This is a standard blind coarse CFO estimator that suppresses the data modulation before spectral peak detection.

### Gardner timing recovery

After matched filtering, the receiver applies Gardner timing recovery:

- cubic interpolation is used to sample early, mid, and late values
- the timing error is computed from the early/late difference and the midpoint sample
- a proportional-integral loop updates the sampling phase and symbol period estimate

In simplified form, the error is:

`e[m] = Re{(x_late - x_early) x_mid*} / energy`

Implementation details:

- the loop is gated by energy percentiles to avoid updating on very weak or very strong outlier samples
- cubic interpolation allows fractional sample timing updates
- the output is a symbol-rate stream

### Fine carrier recovery with Costas loop

Residual phase and frequency error are corrected by a Costas loop:

- the incoming symbol is rotated by the current VCO phase estimate
- a decision-directed phase detector generates the error signal
- a proportional-integral loop updates the phase estimate

For QPSK, the implemented error detector is:

`e = sign(I)Q - sign(Q)I`

For BPSK, the implemented error detector is:

`e = sign(I)Q / power`

Implementation details:

- the loop update is gated using symbol-power percentiles
- this reduces unstable updates outside the active burst region

## 7. Equalization

`equalizer.py` implements an optional short linear equalizer:

- a centered 3-tap complex FIR filter
- trained on the detected Gold header
- applied to the full rotated symbol stream

Training is performed by least squares:

`X h ~= d`

where:

- `X` is the local 3-sample observation matrix built from the received header
- `h` is the unknown equalizer tap vector
- `d` is the known transmitted Gold sequence

The solution is found with `numpy.linalg.lstsq`.

Suggested report wording:

"A short data-aided complex FIR equalizer was trained from the known preamble symbols by least squares and then applied to the full burst to mitigate mild channel distortion."

## 8. Important implementation-specific details worth mentioning

These are small details that make the report more accurate:

- The project uses hard-decision demodulation and hard-decision Viterbi decoding, not soft decisions.
- The modulation path is currently intended for BPSK and QPSK in the synchronization/framing chain.
- The QPSK constellation points are `+/-1 +/- j`, not unit-energy normalized QPSK.
- The Gold sequence is used for three purposes: frame detection, phase-ambiguity resolution, and equalizer training.
- The receive chain is ordered as coarse CFO correction -> matched filter -> Gardner timing recovery -> Costas loop -> Gold detection/rotation -> optional equalization -> hard demodulation.
- The implementation uses concatenated coding: outer Reed-Solomon and inner convolutional coding.

## 9. A compact method-section skeleton

One reasonable structure for the report is:

1. Packet format and framing
2. Channel coding and bit randomization
3. Modulation and pulse shaping
4. Synchronization and frame detection
5. Equalization and symbol decisions
6. Link-layer reliability

You can describe the actual system in one compact paragraph like this:

"Each application message was packed into a fixed 32-byte datagram with a CRC-16 integrity field. The datagram was protected by a concatenated coding chain consisting of an outer Reed-Solomon block code, a deterministic bit interleaver, a synchronous additive scrambler, and an inner terminated convolutional code. The coded bits were mapped to QPSK symbols, framed by a leading and trailing Gold sequence, upsampled, and pulse-shaped by a root-raised-cosine filter before over-the-air transmission with an ADALM-Pluto SDR. At the receiver, coarse carrier offset was estimated by an FFT-based m-th power method, followed by matched filtering, Gardner timing recovery, and Costas-loop carrier recovery. Frame start was detected by normalized correlation with the known Gold sequence, which was also used for phase-ambiguity resolution and optional least-squares training of a short 3-tap complex equalizer. The payload was then hard-demodulated and decoded by Viterbi, descrambling, deinterleaving, and Reed-Solomon decoding before final CRC verification."

## 10. Design motivation and theory links

This section is intended to answer the report requirement that each design choice must be justified by expected operating conditions, not just by the observation that the final implementation works.

### System conditions the design is based on

The implemented link is a short-burst SDR packet system for low-rate text traffic using two ADALM-Pluto radios in the 863-870 MHz band. The relevant conditions are:

- relatively low offered traffic, since messages are short and sporadic rather than continuous media streams
- realistic SDR impairments such as carrier-frequency offset, sampling/timing error, phase ambiguity, AGC variation, and moderate multipath or front-end distortion
- a burst-mode receiver, where synchronization must be reacquired for each packet
- a need for robust packet delivery rather than maximum spectral efficiency

Under these conditions, the design should prioritize robust synchronization, strong packet detection, and error protection over aggressive throughput optimization.

### Data rate

The current configuration uses:

- symbol rate `R_s = 125 ksymbols/s`
- QPSK, giving `2` coded bits per symbol
- sample rate `1 MS/s`
- `8` samples per symbol

This gives a raw coded bit rate of:

`R_b,raw = R_s * log2(M) = 125e3 * 2 = 250 kbit/s`

However, the useful application rate is much lower because of coding, framing, and guard overhead.

With the current configuration:

- application payload per datagram: `23 bytes = 184 bits`
- datagram on air before Reed-Solomon: `32 bytes = 256 bits`
- after Reed-Solomon: `40 bytes = 320 bits`
- after rate-1/3 convolutional coding and byte alignment: `984 coded bits`
- QPSK payload symbols: `984 / 2 = 492 symbols`
- plus Gold framing: `492 + 31 + 31 = 554 symbols`
- burst duration excluding guards: `554 / 125e3 = 4.432 ms`

If the configured guard interval is also included:

- guard length: `64` symbols before and after the burst
- total burst time including guards: `(554 + 64 + 64) / 125e3 = 5.456 ms`

This gives a rough maximum application-layer throughput of:

- about `41.5 kbit/s` excluding guards
- about `33.7 kbit/s` including guards

Design motivation:

- This is comfortably above the requirement for short text messages.
- It leaves margin for strong coding and burst synchronization overhead.
- A higher symbol rate would increase throughput, but would also tighten timing recovery, increase occupied bandwidth, and make the air link more sensitive to imperfect synchronization and SDR front-end limitations.

Suggested wording:

"The symbol rate was chosen to be high enough to support short text packets with low latency, but low enough to preserve robust timing recovery and frequency synchronization on a low-cost SDR platform."

### Bandwidth

The pulse shaping uses a root-raised-cosine filter with roll-off `alpha = 0.5`, so the theoretical occupied passband bandwidth is approximately:

`B ~= (1 + alpha) R_s = 1.5 * 125 kHz = 187.5 kHz`

This gives the main design tradeoff:

- lower roll-off and lower symbol rate reduce spectral occupancy
- higher roll-off gives easier timing recovery and more tolerance to implementation imperfections

The chosen parameters are a practical compromise:

- the theoretical occupied bandwidth remains well below the configured Pluto analog RX bandwidth of `500 kHz`
- the wider analog front-end bandwidth gives tolerance to residual carrier offset, analog filter mismatch, burst transients, and non-ideal SDR filtering
- software RRC pulse shaping still controls the actual transmitted spectrum

Design motivation:

- The project is not bandwidth-limited in the same way as a dense multi-user commercial system.
- It is therefore reasonable to spend some bandwidth margin on synchronization robustness.
- This is consistent with many practical packet-radio and telemetry-style links, where excess bandwidth is accepted to simplify timing recovery and pulse-shaping implementation.

### Required BER and packet reliability

For this system, the most relevant performance metric is not only uncoded BER, but post-decoder packet reliability:

- the packets are short
- the payload is human-readable text
- a single wrong byte can corrupt the meaning of the message
- CRC and optional retransmission operate on the packet level

For a `32-byte` datagram, the uncoded packet error probability grows quickly with BER. Ignoring coding and CRC details for a moment:

`PER ~= 1 - (1 - BER)^256`

Examples:

- if `BER = 10^-3`, then `PER` is roughly `22 %`
- if `BER = 10^-4`, then `PER` is roughly `2.5 %`
- if `BER = 4 * 10^-5`, then `PER` is roughly `1 %`

This shows why packet-oriented systems usually need coding, CRC, and possibly retransmission even when the raw BER looks fairly small.

Design motivation:

- The practical target should be low residual BER after decoding and a sufficiently low packet loss rate that text messages arrive correctly with little or no retransmission.
- Because the source traffic is short and delay-tolerant, it is acceptable to spend substantial overhead on reducing the post-decoder error rate.
- In this project, BER and datagram error rate are both meaningful, and the repository already contains a `bit_error_test_script.py` that reflects this dual view.

Suggested wording:

"The required error performance was driven by packet integrity rather than by a raw BER figure alone. Since even a small residual BER can lead to a high packet error probability for a 32-byte datagram, the system was designed to minimize post-decoder packet failure through concatenated coding, CRC verification, and optional retransmission."

### Source coding

No source coding or compression is used in the current design.

This should be motivated explicitly:

- the source is short text messages, not audio, image, or continuous sensor data
- the offered source rate is already low
- short messages compress poorly in many cases and can even expand once framing and dictionary overhead are included
- compression would add implementation complexity and potential error sensitivity without giving a major quality-of-service benefit

Design motivation:

- The chosen application does not require source coding to meet the available data rate.
- The simpler design is justified because the channel rate, even after heavy overhead, is sufficient for short text communication.
- This is preferable to adding a compression stage that would complicate framing and error handling while providing limited gain for short packets.

Suggested wording:

"No source coding was applied because the target traffic consists of short text messages with a low offered rate. The available net data rate was sufficient for the desired quality of service, so adding compression would mainly increase complexity rather than provide a meaningful system-level benefit."

### Channel coding

The coding chain combines:

- outer Reed-Solomon coding
- bit interleaving
- inner rate-1/3 convolutional coding

This is a classical concatenated-coding architecture. The theoretical motivation is that different error-control codes are strong against different error types:

- convolutional coding with Viterbi decoding is effective against random bit errors
- Reed-Solomon coding is effective against clustered symbol or byte errors
- interleaving converts localized error bursts into a more distributed pattern before the outer decoder sees them

This type of layered design is widely used in practical communications systems and has long been common in satellite links, telemetry links, and older packet-radio systems because it offers good robustness with moderate implementation complexity.

Design motivation:

- the channel impairments in an SDR burst receiver are not purely random; synchronization slips, fades, and phase errors can create short error bursts
- using only a convolutional code would leave the system more vulnerable to clustered decoder failures
- using only Reed-Solomon would give weaker protection against the dense bit errors that occur before symbol decisions stabilize

The chosen structure is therefore appropriate for the mixed random-plus-burst error behavior expected in this project.

### Interleaver

The interleaver operates over the full coded frame length after Reed-Solomon and before scrambling and convolutional coding.

The motivation should be tied to channel behavior:

- burst-mode SDR reception can produce error clusters when timing is briefly wrong, when phase recovery slips, or when a short fade hits part of the burst
- Reed-Solomon works best when those clustered errors are spread across more symbols instead of appearing contiguously

Why a full-frame interleaver is reasonable here:

- the packets are short
- the full frame is only a few hundred bits before the convolutional code
- latency added by interleaving is negligible for this application
- a single whole-frame permutation is simple to implement and easy for TX and RX to reproduce deterministically

Design motivation:

- The interleaver length should be long enough to spread likely error bursts over the frame.
- Since the entire datagram is short, using the full frame as the interleaving span is a natural low-complexity choice.

Suggested wording:

"A frame-length bit interleaver was used because the expected channel errors are partly burst-like rather than independent from bit to bit. Since the packets are short, full-frame interleaving provides useful burst decorrelation with negligible additional latency."

### Modulation

The implemented physical layer uses QPSK by default, while BPSK remains available.

The theoretical tradeoff is standard:

- BPSK is more power-efficient and simpler to recover
- QPSK doubles the coded bit rate for the same symbol rate and bandwidth
- higher-order constellations would increase spectral efficiency but also require higher SNR and tighter synchronization

Design motivation for QPSK:

- It gives a clear improvement in throughput over BPSK.
- It still retains the robustness and simple synchronization properties of low-order PSK.
- It is well matched to non-data-aided timing recovery and Costas-loop carrier recovery.
- It is a common practical choice in robust burst and packet links for exactly this reason.

The report can also mention that keeping BPSK support is useful for fallback testing and comparison under worse channel conditions.

### Demodulation

The receiver uses hard-decision slicing and hard-decision Viterbi decoding.

This choice should be motivated as a complexity-performance tradeoff:

- hard decisions are simpler to implement and debug
- the current receiver already has significant burst-synchronization complexity
- hard decisions are computationally cheaper and easier to keep deterministic
- soft decisions would likely improve coding gain, but would require reliability metrics from the demodulator and a different decoder interface

Design motivation:

- For a student SDR project with short packets, hard decisions are a reasonable compromise between performance and implementation complexity.
- The use of strong coding, CRC, and optional retransmission partly compensates for the performance loss relative to soft-decision decoding.

### Synchronization

Synchronization is one of the most strongly motivated parts of the design because the system is burst-based and runs on separate SDR oscillators.

The chosen chain is:

- coarse FFT-based m-th power carrier-offset estimation
- matched filtering
- Gardner timing recovery
- Costas-loop fine carrier recovery
- Gold-code frame detection and phase ambiguity resolution

Each block has a separate role:

- coarse carrier estimation removes large CFO before symbol decisions become reliable
- matched filtering maximizes SNR at the sampling instant for the chosen pulse shape
- Gardner timing recovery is a standard non-data-aided timing loop for PSK/QPSK waveforms
- Costas loops are classical carrier-recovery methods for PSK signals
- a known Gold header provides robust burst detection and a reference for phase rotation and equalizer training

This is closely related to the structure used in many classical digital receivers and burst modems: pulse shaping and matched filtering, timing recovery, carrier recovery, preamble correlation, and then decoding.

Design motivation:

- Separate Pluto devices can have non-negligible frequency and phase offsets.
- Burst reception means the receiver cannot assume continuous tracking across packets.
- A known header is therefore needed for fast packet detection and reliable ambiguity resolution.
- Non-data-aided timing and carrier loops are attractive because they do not require a long training sequence in the payload itself.

Suggested wording:

"The synchronization chain was selected to address the dominant impairments of a burst-mode SDR link: carrier offset, timing error, phase ambiguity, and uncertain packet start. The chosen combination of FFT-based coarse correction, Gardner timing recovery, Costas-loop carrier recovery, and Gold-sequence correlation is well aligned with standard digital-receiver theory and is widely representative of practical burst receivers."

### Why this overall architecture is reasonable

The most important overall argument is that the project is a short-packet SDR link, not a high-throughput commercial modem. Under those conditions, the design should favor:

- robust synchronization over very high spectral efficiency
- strong packet integrity over minimum redundancy
- low and understandable implementation complexity over theoretically optimal but fragile solutions

That is exactly what the current design does:

- low-order PSK instead of dense constellations
- matched filtering and standard synchronization loops instead of highly specialized estimators
- concatenated coding and CRC instead of relying on uncoded BER performance
- simple source handling instead of compression
- short bursts with strong framing instead of continuous-stream assumptions

This is the core design motivation to emphasize throughout the report: the chosen components are appropriate because they should work sufficiently well for low-rate burst communication under realistic SDR impairments, while remaining implementable, testable, and explainable.

## 11. Small caveats to watch while writing

- If you state the scrambler polynomial explicitly, verify the exact polynomial convention you want to present. The code is LFSR-based, but tap indexing and polynomial notation can be described in multiple equivalent ways.
- If you state the exact Reed-Solomon correction capability, phrase it in terms of the configured `RSCodec(8)` implementation rather than the outdated comments in `forward_error_correction.py`.
- If you describe the expected payload symbol count, note that the implementation packs coded bits back into bytes before modulation, so byte alignment matters slightly.
