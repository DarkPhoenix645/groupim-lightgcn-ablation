# Weeplaces is the reproduction baseline; the other three datasets are reconstructed variants

Author-released Weeplaces is the only dataset on which a number may be compared to GroupIM Table 2. Yelp, Gowalla, and Douban stay in the study. Their files are frozen reconstructed variants: Yin Yelp-LA with prebuilt groups, Gowalla outings as separate groups, and Douban events as items. A miss against Table 1 is recorded in [table1-comparison.md](../table1-comparison.md) and [dataset-notes/](../dataset-notes/). It does not, by itself, mean the preprocessing is wrong, and it does not block the Weeplaces encoder comparison. The variants are not dropped. They are not described as the paper's benchmarks.

Supersedes [ADR-0005](0005-four-datasets-never-drop.md).

**Status:** accepted

**Considered options:** keep rebuilding with §5.1 until Table 1 matches. Rejected: the public sources do not yield those counts, and holding the encoder comparison for that match does not make the variants into the original benchmarks.
