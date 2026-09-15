# results/

Driver output; git-ignored apart from this file.

```text
results/
├── splits/          split assignments shared between Python and R
├── *.json           summaries plus every per-replicate value under "raw"
├── *.ckpt.pkl       checkpoints, written after each cell; see --resume
├── *.tex            table fragments to paste into the manuscript
└── cox_lp_*.csv     held-out Cox linear predictors from the R scripts
```

The `.json` files are the ones worth backing up: they hold the raw
per-replicate numbers, so any statistic not computed at run time can be derived
later without repeating the grid.

Figure panels are written to `manuscript/images/` when that directory exists
and here otherwise.
