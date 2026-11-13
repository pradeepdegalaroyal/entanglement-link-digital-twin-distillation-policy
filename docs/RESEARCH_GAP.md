# Research gap: closing the loop between link monitoring and distillation decisions

## State of the art in the supplied literature

References are given as author, year and digital object identifier (DOI).

Entanglement-based networks are the central object of the quantum internet roadmap of Wehner et al., 2018 (doi:10.1126/science.aam9288), which orders the field into stages of increasing functionality and names fidelity-limited entanglement distribution as the bottleneck of the early stages. Quantum communication is also listed as one enabling technology of sixth-generation (6G) networks (Zhang et al., 2019, doi:10.1109/MVT.2019.2921208; Chowdhury et al., 2020, doi:10.1109/OJCOMS.2020.3010270), but there it appears as a feature list, not as a quantitative link model.

The most recent evidence on how such networks should be *engineered* comes from the systematic review of Elsokary et al., 2027 (doi:10.1016/j.dte.2026.100149). It sorts the literature into digital models, digital shadows (one-way data flow from the hardware) and early digital twins, and concludes that "true digital twins, with continuous synchronisation between physical quantum networks and their digital counterparts, remain undeveloped". A scientometric survey (Sharma and Sood, 2027, doi:10.1016/j.cosrev.2026.101092) confirms scalable quantum networking and hybrid quantum-classical control as growing research fronts.

Three recent papers supply methodological ingredients without addressing links. Mohammed, 2027 (doi:10.1016/j.cpc.2026.110477) shows that a fixed sampling budget is statistically inadequate and that observable-specific sequential stopping with a minimum-sample guard removes both under- and over-sampling. Raigada-García et al., 2027 (doi:10.1016/j.future.2026.108903) treat quantum workloads as schedulable units in a hybrid classical-quantum runtime. Gao et al., 2027 (doi:10.1016/j.epsr.2026.114409) illustrate that estimation under weak, noisy features is the common difficulty in state localisation problems.

## Related work table

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Wehner, Elkouss, Hanson (doi:10.1126/science.aam9288) | 2018 | landmark | Stages of a quantum internet; entanglement distribution as the enabling primitive | A quantitative control loop that keeps link fidelity usable while the hardware drifts |
| Zhang et al. (doi:10.1109/MVT.2019.2921208) | 2019 | landmark | 6G vision with quantum communication as one key technology | Any link-level fidelity or distillation model |
| Chowdhury et al. (doi:10.1109/OJCOMS.2020.3010270) | 2020 | landmark | 6G requirements and research directions, including quantum communications | Monitoring cost, decision-making under drift |
| Elsokary et al. (doi:10.1016/j.dte.2026.100149) | 2027 | recent (review) | Taxonomy digital model / shadow / twin for quantum networks; finds no true twin | An implemented, measured twin; a comparison against the shadow it is supposed to supersede |
| Sharma and Sood (doi:10.1016/j.cosrev.2026.101092) | 2027 | recent (review) | Scientometric map of quantum computing and networking fronts | Any algorithmic or numerical result on links |
| Mohammed (doi:10.1016/j.cpc.2026.110477) | 2027 | recent | Observable-dependent adaptive stopping with a minimum-sample guard for trajectory simulation | Use of the stopping idea for *physical* pair consumption on a drifting link |
| Raigada-García, Jorba, Iserte (doi:10.1016/j.future.2026.108903) | 2027 | recent | Wave-based dispatch of circuit fragments across quantum and classical back ends | Entanglement links, drift, fidelity thresholds |
| Gao et al. (doi:10.1016/j.epsr.2026.114409) | 2027 | recent | Quantum-inspired optimiser for fault localisation under noisy features | Quantum states; sequential measurement cost |
| Övgün (doi:10.1142/S0219887826501975) | 2027 | recent | Detector-model universality of horizon-brightened acceleration radiation | Networks or engineering decisions (cited for the quantum-optics detector viewpoint only) |
| Zhou et al. (doi:10.15302/frontphys.2027.042203) | 2027 | adjacent | Nanofiber near-field collection of colour-centre fluorescence, 0.22 % and 0.18 % peak efficiency | Network-level consequences of low collection efficiency (it motivates scarce pairs) |

## The gap

Reading the table across roles: the landmark roadmap says entanglement distribution must work at scale, the review says nobody has built the synchronised digital counterpart that would let operators run it, and the recent methodological papers say that how many samples to spend should be decided online and per quantity of interest. No supplied paper measures what a synchronised twin of an entanglement link is *worth* once the pairs spent on keeping it synchronised are charged to the link itself. That charge is large: each monitoring pair is a pair that is not delivered.

**Concrete question.** For a dephasing-biased entanglement link whose fidelity and noise bias drift, at which drift speed and pair budget does a particle-filter digital twin with decision-aware sequential monitoring deliver more threshold-meeting entanglement than (a) a periodic-recalibration digital shadow with its schedule tuned, (b) a static digital model and (c) the best fixed distillation policy, when monitoring pairs are subtracted from the delivered pairs?

**Novelty in one sentence.** This is a measured, cost-charged comparison of digital model, shadow and twin for an entanglement-distribution link, with a pre-posterior monitoring rule that adapts the Mohammed-style sequential stopping idea to pairs that physically disappear.

## Hypotheses and success metric

- **H1 (drift regime).** At the headline operating point (fast drift, 96 raw pairs per cycle) the twin has higher *efficiency* than the strongest digital shadow, where efficiency is delivered threshold-meeting entanglement divided by the oracle value that knows the true state and spends nothing on monitoring. Success: paired mean difference with a bootstrap 95 % interval above zero over 10 seeds.
- **H2 (phase boundary).** The twin advantage is not universal: it vanishes at small pair budgets, where monitoring is too costly, and when the drift is slow enough that a fixed policy is already near-optimal. Success: a region of the (volatility, budget) plane where the twin is not ahead of the strongest baseline, reported as such.
- **H3 (what matters).** Both filter memory and cost-aware stopping contribute; removing either lowers efficiency. Falsified if an ablation matches the full twin within seed noise.

## Scale and honesty of the setting

Everything is classical simulation of Bell-diagonal two-qubit states on a laptop: no quantum advantage is claimed, nothing here is a hardware measurement, and the drift process is an assumed Ornstein-Uhlenbeck-plus-jumps model. The recurrences are validated against an explicit four-qubit density-matrix simulation. The value of the study is the decision structure and the measured break-even map, which transfer to hardware once the drift model is fitted to link data.
