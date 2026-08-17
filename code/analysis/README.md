# JMS analysis code

This directory contains the analysis path used for the Journal of Medical Systems
version of TedFace-Flow. It covers the eight rectus muscles only: bilateral superior,
inferior, medial, and lateral rectus muscles.

## Script-to-manuscript mapping

- `morphometry.py` defines the shared mask-based measurements. Muscle volume is the
  labelled voxel volume. MCSA is the maximum binned cross-sectional area orthogonal
  to the principal muscle axis. The axis is fitted to the largest connected component.
  Measurements below the prespecified 50 mm3 volume threshold are missing, not zero.
- `extract_generated_morphometry.py` applies the shared rule to masks segmented from
  generated proxies.
- `extract_gt_morphometry.py` applies the identical rule to CT-derived reference masks.
- `proxy_group_analysis.py` evaluates one inference run, including the single-seed
  component ablations and conditioning controls.
- `multiseed_formal_reanalysis.py` implements the reported three-seed primary analysis,
  the at-least-two-seed sensitivity analysis, 10,000 bootstrap resamples, Welch tests,
  Hedges' g, 10,000 group-label permutations, Benjamini-Hochberg correction, and the
  10,000 no-self-match patient-correspondence control.

## Reference-group and missing-data rules

For each muscle and measure, the lower and higher reference groups are defined by the
30th and 70th percentiles of the reference CT measurement among patients with a valid
generated measurement under the analysis rule being evaluated. Generated measurements
are outcomes only and never define group membership.

The primary multi-seed rule requires all three seeds (2026, 42, and 1024) to be valid
before averaging at patient level. The sensitivity rule averages available values when
at least two seeds are valid. Segmentation failures remain missing and are never
zero-imputed. The patient is the independent statistical unit.

Benjamini-Hochberg adjustment is performed separately for the eight MCSA comparisons
and the eight volume comparisons, and separately for each p-value family. Every
patient-correspondence replicate uses one common 74-patient no-self-match derangement
across all eight muscles and both measures.

## Inputs

Patient-level images, masks, photographs, and clinical records are not included. The
scripts accept paths through command-line arguments; no institutional server paths or
patient identifiers are hard-coded in this release.
