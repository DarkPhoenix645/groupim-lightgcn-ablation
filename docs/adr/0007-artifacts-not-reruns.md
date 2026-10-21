# One config → one CSV row; dump per-group artifacts so analysis never retrains

The only GPU is one RTX 3080 (10 GB). The ~450 runs / 80–120 GPU hours figure was sized for a 4080 and has to be remeasured here. Plan on 2–3 concurrent jobs. Every run must write a results-CSV row plus per-group test metrics, size, coherence, diversity, and per-member discriminator scores. Analysis (size bins, MI-variation, short-history hypothesis, λ plots) reads those files. Retraining to answer a plot question is a process failure.

**Status:** accepted
