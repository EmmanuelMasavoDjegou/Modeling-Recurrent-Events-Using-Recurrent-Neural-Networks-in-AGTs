# results/

Driver output. Everything here except this file and `.gitkeep` is git-ignored.

```text
results/
├── splits/          split and cross-validation fold assignments, shared with R
├── *.json           summaries, plus every per-replicate value under "raw"
├── *.ckpt.pkl       checkpoints written after each cell; see --resume
├── *.tex            table fragments for the manuscript
└── cox_lp_*.csv     Cox linear predictors exported by the R scripts
```

The `.json` files are the ones worth keeping: they hold the raw per-replicate
numbers, so a statistic not computed at run time can be derived afterwards
without repeating the grid. `experiments/paired_capacity.py` is an example.

Figure panels are written to `manuscript/images/` when that directory exists
and here otherwise.
