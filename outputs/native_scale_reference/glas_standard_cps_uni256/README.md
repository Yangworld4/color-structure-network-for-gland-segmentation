# E4 Native-Scale Reference-Guided CPS Feasibility Diagnostic

This is a closed, offline, no-training diagnostic using only the fixed UNI-256
native 16x16 evidence, standard CPS checkpoints, and validation GT for offline
evaluation. A token is a UNI contradiction when the two pooled Student
predictions agree but the UNI semantic decision disagrees with their consensus.
GT never enters that signal; it is used only to compute token-footprint error
ratios, coverage, direction accuracy, event coverage, and counterfactual
descriptive steps.

The fixed UNI-256 memory is
`checkpoints/glas_labeled10_seed24/reference_memory_uni256.npz`, built from the
same `labeled_10.csv` source with the project top-K cosine/memory rules. No
margin interpolation, pixel pseudo-label, R, CPS loss, or training is used.
The direction dead zone is fixed at 0.05 and hypothetical step sizes are
fixed at 0.1 and 0.25; neither is selected using validation results.
