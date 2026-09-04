# RNN-AGT: A Recurrent Neural Network Framework for Accelerated Gap-Time Modeling via Gehan-Type Rank Loss.

> **Before running anything:** four defects in the original implementation were
> corrected in this release, two of which changed published numbers. See
> [CORRECTIONS.md](CORRECTIONS.md), and run `python experiments/run_diagnostics.py`
> to verify your installation.

---

## Repository structure

```text
.
├── rnn_agt/                     # the package: one definition of everything
│   ├── seeds.py                 # four independent RNG streams from one master seed
│   ├── data.py                  # data-generating mechanisms, censoring, tau calibration
│   ├── models.py                # AFTWRS, NNAFT, RNNAGT (the ablation ladder)
│   ├── losses.py                # Gehan-WRS objective, exact and subsampled
│   ├── sampling.py              # pair sampler with explicit inclusion probabilities
│   ├── metrics.py               # vectorised IPCW C-index and AMSE
│   ├── train.py                 # one trainer, all three model classes
│   ├── evaluation.py            # repeated splits, k-fold CV, paired differences
│   ├── cox_bridge.py            # R/Python bridge for the Cox comparators
│   ├── latex.py                 # emits table rows for the manuscript
│   └── diagnostics.py           # correctness checks
│
├── experiments/                 # command-line drivers, one per manuscript table
│   ├── run_diagnostics.py       # run this first
│   ├── run_simulation_tables.py         Tables 1-3
│   ├── run_dependence_robustness.py     Table 5
│   ├── run_benchmark_rows.py            Table 7
│   ├── run_ablation_and_splits.py       Tables 8 and 9
│   └── run_capacity_sweep.py            Table 10
│
├── simulation/                  # exploratory notebooks
│   ├── model_demo.ipynb                 core demonstration of the three models
│   ├── defect_impact.ipynb              measures the four corrected defects
│   ├── subsampling_sensitivity.ipynb    Table 4
│   ├── high_dimensional.ipynb           behaviour as p grows
│   └── figures.ipynb                    manuscript figures
│
├── application/                 # real-data analysis
│   ├── real_data_analysis.ipynb         current analysis (Tables 6-9)
│   ├── split_sensitivity.ipynb          what the single split cost
│   ├── Dataset1_Data_Preprocessing.R            colorectal readmission -> data/crc.csv
│   ├── Dataset1_Classical_Recurrent_Event_Models.R
│   ├── Dataset2_Data_Preprocessing.R            CGD -> data/cgd.csv
│   └── Dataset2_Classical_Recurrent_Event_Models.R
│
├── literature_review/           # background reading
├── data/                        # generated CSVs (git-ignored)
└── results/                     # tables, figures, JSON (git-ignored)
```

**All paths are relative to the repository root.** Run scripts from there
(`python experiments/...`, `Rscript application/...`); notebooks add the root
to `sys.path` themselves.

### Why the structure changed

The model, loss, sampler, metrics and data generation used to be defined
**inline in each notebook**, with the same block copy-pasted six times. That is
why the defects in CORRECTIONS.md survived: fixing one copy left five
untouched, and the copies had drifted apart. There is now one definition of
each, in `rnn_agt/`, and the notebooks only set up experiments and report them.

### Datasets

**Dataset 1 is colorectal cancer readmission** (`frailtypack::readmission`,
n=403). **Dataset 2 is chronic granulomatous disease** (`survival::cgd`,
n=128).

---

## Which script produces which table

| Table | Contents | Produced by |
|---|---|---|
| 1 | Interaction mean function | `experiments/run_simulation_tables.py` |
| 2 | GAM-type nonlinear mean function | same |
| 3 | Linear mean function | same |
| 4 | Sub-sampling sensitivity | `simulation/subsampling_sensitivity.ipynb` |
| 5 | Dependence-mechanism robustness | `experiments/run_dependence_robustness.py` |
| 6 | Real-data train/test metrics | `application/real_data_analysis.ipynb` |
| 7 | Benchmark, single fixed split | `experiments/run_benchmark_rows.py` |
| 8 | Ablation ladder | `experiments/run_ablation_and_splits.py` |
| 9 | Repeated splits and 5-fold CV | same |
| 10 | Capacity sweep | `experiments/run_capacity_sweep.py` |

Every driver writes a `.json` with raw per-replicate numbers and a `.tex`
fragment that pastes over the corresponding placeholder cells in the
manuscript.

---

## Getting started

```bash
pip install -r requirements.txt
python experiments/run_diagnostics.py        # five checks, all should pass
```

### Simulations

```bash
python experiments/run_simulation_tables.py --replicates 500      # Tables 1-3
python experiments/run_dependence_robustness.py --replicates 500  # Table 5
```

Add `--quick` to either for a two-replicate pipeline test.

### Real data

The analysis runs in three stages, because the Cox comparators are fitted in R
and must see the same splits as the neural models.

```bash
# 1. preprocess -> data/crc.csv, data/cgd.csv
Rscript application/Dataset1_Data_Preprocessing.R
Rscript application/Dataset2_Data_Preprocessing.R

# 2. Python fits the AFT models and writes split assignments
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd.csv --crc data/crc.csv \
    --splits 200 --splits-out results/splits

# 3. R fits the Cox models on those same splits, exporting held-out
#    linear predictors (not concordance indices)
Rscript application/Dataset1_Classical_Recurrent_Event_Models.R \
    --data data/crc.csv --splits results/splits/splits_crc.csv \
    --out results/cox_lp_crc.csv
Rscript application/Dataset2_Classical_Recurrent_Event_Models.R \
    --data data/cgd.csv --splits results/splits/splits_cgd.csv \
    --out results/cox_lp_cgd.csv

# 4. re-run step 2 with the Cox predictors to fill Table 9
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd.csv --crc data/crc.csv \
    --splits 200 --splits-out results/splits \
    --cox-lp-cgd results/cox_lp_cgd.csv \
    --cox-lp-crc results/cox_lp_crc.csv

python experiments/run_capacity_sweep.py \
    --cgd data/cgd.csv --crc data/crc.csv --splits 200   # Table 10
```

Exporting linear predictors rather than a concordance computed in R is
deliberate: there is exactly one concordance implementation in the project, so
the two model families cannot drift onto different estimands. Python negates
the linear predictor once (log-hazard scale to log gap-time scale); never
negate it in R as well.

---

## Notes on use

**AFT-WRS needs a larger learning rate.** Three parameters against the GRU's
tens of thousands means it barely moves at `3e-4`; the drivers default it to
`1e-2` via `--lr-linear`. An undertrained comparator would flatter RNN-AGT for
the wrong reason, so confirm it has converged before reporting it.

**NN-AFT is parameter-matched to the GRU** via `models.matched_mlp_width`, so a
gap between those two rungs cannot be attributed to one model simply being
larger.

**Win rate is reported alongside every paired difference.** A method can carry
a positive mean increment while losing on 40% of splits; those are different
claims.

---

## Citation

See `CITATION.cff`. Licensed under MIT (`LICENSE`).
