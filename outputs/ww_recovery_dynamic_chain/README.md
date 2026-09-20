# WW Recovery Dynamic Chain Diagnostic

Offline deterministic validation diagnostic; no optimizer, backward pass, or checkpoint modification.

Common epochs: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]`. State definition: `CC=A=B=GT`, `WW=A=B!=GT`, `DIS=A!=B`, with `p>=0.5`.
Adaptive contradiction is recomputed from frozen UNI-256 reference evidence and is never defined from GT.

Outcome: **Outcome B** — Adaptive suppression breaks wrong consensus but produces excessive persistent disagreement.

Prediction archives are under `predictions/`; CSVs contain global rates, conditional transitions, three-step chains, targeted/matched cohorts, D quartiles, and disagreement balance.
