# Method

## System model

A two-node link generates `N` raw entangled pairs per cycle. Each pair is a Bell-diagonal state with weights `p = (A, B, C, D)` on the Bell states (Phi+, Psi-, Psi+, Phi-). The link is dephasing biased: `A = F`, the Phi- weight is `(1 - F) w`, and Psi+ and Psi- share the remainder. The latent coordinates `z = (z1, z2)` set `F = 0.5 + 0.5 sigmoid(z1)` and `w = sigmoid(z2)`.

**Drift.** `z` follows an Ornstein-Uhlenbeck process (correlation 0.95 per cycle, stationary mean `(0, 1)`) with volatility `sigma` on `z1` (default 0.2) and 0.05 on `z2`, plus a jump of size 0.8 in `z1` with probability 0.02 per cycle. This stands in for slow thermal and alignment drift plus occasional step events such as a lock loss. It is an assumption, not a fitted hardware model.

**Actions.** Six distillation policies: raw pairs, one or two rounds of the Bennett-Brassard-Popescu-Schumacher-Smolin-Wootters (BBPSSW) protocol, and one to three rounds of the Deutsch-Ekert-Jozsa-Macchiavello-Popescu-Sanpera (DEJMPS) protocol. Both are applied as exact Bell-diagonal recurrences. Before every round both input copies are mixed with the maximally mixed state with weight `q` (local gate noise, default 0.02). A round turns two inputs into one output with its success probability, so it yields `n/2` outputs per input.

**Value.** A delivered pair counts only if its fidelity reaches the application threshold `F_req = 0.90`. The value of a cycle is the number of threshold-meeting output pairs, `(N - m) x v(a, state)`, where `m` is the number of raw pairs spent on monitoring and `v` the expected outputs per raw pair. The threshold makes the loss asymmetric: under-distilling gives zero, over-distilling wastes pairs.

**Efficiency (the metric).** Total delivered value divided by the total value of an *oracle* that knows the true state every cycle and spends nothing on monitoring. Also reported: monitoring pairs per cycle and the outage rate (cycles with zero delivered value although the oracle delivers some).

**Monitoring.** Spending `m` pairs measures `m/3` pairs in each of the Pauli ZZ, XX and YY correlation bases (matched local measurements with classical communication). Each correlator count is binomial; the three correlators determine `p` exactly in expectation. A per-qubit readout error `e` attenuates every correlator by `(1 - 2e)^2`.

## Methods compared

| Method | Role | Rule |
|---|---|---|
| Oracle | upper bound | true state known, no monitoring cost |
| Raw pairs | classical baseline | never distil |
| Fixed policy | standard quantum baseline | one fixed protocol (the textbook DEJMPS, two rounds) and the best fixed action chosen in hindsight |
| Static digital model | digital model | one calibration at cycle 0, never updated |
| Digital shadow | digital shadow | recalibrate every `P` cycles with `M` pairs, plug-in best action in between; grid `P in {1,2,4,8,16,32}`, `M in {12,24,48,96}`; reported both tuned on three disjoint seeds and best in hindsight on the test seeds (the stricter, baseline-friendly choice) |
| **Digital twin** | proposed | particle filter over `z` with the drift model, Bayes action under the posterior, pre-posterior monitoring rule |

**Twin details.** 400 particles are propagated with the drift model (and jumps), weighted by the binomial likelihood of the monitoring counts, and resampled every cycle. The action maximises posterior expected value. *Monitoring rule:* with `used` pairs already spent this cycle, the twin simulates monitoring `s` more pairs for `s in {6, 12, 24}` by drawing states from the posterior, simulating counts, updating and scoring the resulting Bayes action on the drawn state. It continues, one six-pair batch at a time, only while the best expected net gain `(N - used - s) E[value after] - (N - used) E[value now]` is positive, after a minimum of `m_min = 6` pairs and up to `m_max = 60`. The minimum guard is the analogue of the minimum-trajectory safeguard in the adaptive-stopping literature.

## Ablation (one component removed at a time)

1. no filter memory (flat prior every cycle), 2. fixed monitoring of 12 pairs per cycle (no sequential rule), 3. no minimum guard (`m_min = 0`), 4. no jump component in the twin's drift model.

## Sweeps

- **Phase diagram:** drift volatility `sigma in {0.05, 0.1, 0.2, 0.3, 0.4}` by pairs per cycle `N in {24, 48, 96, 192, 384}`; twin minus the strongest baseline (best of all fixed and shadow configurations in hindsight).
- **Noise and robustness:** gate noise `q in {0, 0.01, 0.02, 0.04, 0.08}`; readout error `e in {0, 0.02, 0.05, 0.1}` with estimators that know `e` and with estimators that wrongly assume `e = 0`; twin drift-model mismatch (assumed over true volatility `in {0.25, 0.5, 1, 2, 4}`).

## Seeds and validation

Headline: 10 test seeds of 150 cycles, three disjoint tuning seeds for the shadow. Sweeps: 5 seeds of 100 cycles. All methods see the same latent trajectory per seed (common random numbers); measurement noise is independently seeded per method. Statistics are paired differences with a 10,000-resample bootstrap over seeds. The closed-form recurrences are tested against an explicit four-qubit density-matrix simulation (controlled-NOT, local rotations, projective measurement) in `tests/test_bell.py`.
