# PEBFMT* Experimental Data

This repository contains the run-level simulation, ablation, and robot data
associated with the manuscript **“PEBFMT*: a path planning strategy for robotic
arms under limited planning budgets.”**

## Contents

- `data/main-study/`: 1,000 main-study runs and normalized one-second traces.
- `data/ablation/`: 1,000 ablation runs and normalized one-second traces.
- `data/robot/`: 120 Sawyer pick-place-reset trials.
- `results/`: the numerical sources for Tables 3 and 4 and the robot summary.
- `scripts/recompute_statistics.py`: verification of the released data and
  manuscript statistics.
- `src/README.txt`: algorithm source-code availability notice.

The original internal identifiers `DAFMT_star` and `DAFMTkConfigDefault` are
retained in the data; `PEBFMT*` is the corresponding manuscript-facing name.
Failed, timed-out, approximate-only, and target-non-reaching runs remain in the
released tables.

Raw benchmark logs, build logs, host information, local paths, and compiled
files are not included.

## Verify the experimental data

Python 3.9 or later is sufficient; no third-party package is required.

```bash
python3 scripts/recompute_statistics.py
```

The checker validates row counts, run identifiers, outcome fields, target-hit
statistics, confidence intervals, robot-stage totals, and the reported
main-study, ablation, and robot summaries.

## License

The released data are distributed under CC BY 4.0 as stated in
`DATA_LICENSE.md`.
