# One config → one CSV row; dump per-group artifacts so analysis never retrains

The only GPU is one RTX 3080 (10 GB). A full Weeplaces run takes 2.0 hours on this card, measured for the MLP encoder and for LightGCN K=3. The locked queue is 94 GPU hours of Weeplaces runs ([scope.md](../scope.md)). One job was timed. A second Weeplaces job may share the card only after a paired run finishes. Every run must write a results-CSV row plus per-group test metrics, size, coherence, diversity, and per-member discriminator scores. Analysis (size bins, MI-variation, short-history hypothesis, λ plots) reads those files. Retraining to answer a plot question is a process failure.

**Status:** accepted
