# One config → one CSV row; dump per-group artifacts so analysis never retrains

The 4080 budget is ~450 runs / 80–120 GPU hours, 2–3 concurrent jobs. Every run must write a results-CSV row plus per-group test metrics, size, coherence, diversity, and per-member discriminator scores. Analysis (size bins, MI-variation, short-history hypothesis, λ plots) reads those files. Retraining to answer a plot question is a process failure.

**Status:** accepted
