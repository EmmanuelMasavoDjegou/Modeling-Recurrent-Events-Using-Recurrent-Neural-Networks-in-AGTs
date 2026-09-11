# RNN-AGT

A recurrent neural network framework for semiparametric accelerated gap-time
modeling of recurrent event data, estimated with a Gehan-type weighted
risk-set (WRS) rank loss.

Code accompanying *RNN-AGT: A Recurrent Neural Network Framework for
Accelerated Gap-Time Modeling via Gehan-Type Rank Loss* by Emmanuel Masavo
Djegou, Akim Adekpedjou and Xuerong Meggie Wen.

---

## Method

RNN-AGT extends the linear accelerated gap-time model to allow nonlinear
covariate effects and history-dependent event dynamics. The linear predictor is
replaced by a stacked GRU whose hidden state carries the subject's event
history, and the model is estimated with the Gehan-type WRS objective

```
L(w) = (1/n) sum_i sum_j sum_l sum_k [ delta_ij / (K_i* K_l*) ] [ e_ij - e_lk ]^-
```

where `e_ij = log(G_ij) - mu_ij(w)` is the residual on the observed log-gap
scale, `K_i* = max(K_i, 1)`, and `[a]^- = max(0, -a)`. The subject-level weights
`1/(K_i* K_l*)` equalise each subject's contribution regardless of how many
events they accrued, which is what handles the dependent censoring induced by
the recurrent observation structure.

Because the objective is a sum over pairs, it is estimated from an
importance-weighted subsample of pairs, giving a conditionally unbiased
stochastic subgradient of the full loss.

### The three model classes

The package implements a ladder in which each rung adds exactly one capability,
so the contribution of each can be measured separately.

| Model | Nonlinear | History | Predictor |
|---|---|---|---|
| `AFTWRS` | no | no | `theta' z_i` |
| `NNAFT` | yes | no | `MLP(z_i)` |
| `RNNAGT` | yes | yes | `GRU(z_i, prior log-gaps)` |

All three train under the identical objective, sampler and optimizer, so a
difference between adjacent rungs is attributable to the capability that rung
adds. `AFTWRS` versus `NNAFT` isolates nonlinearity; `NNAFT` versus `RNNAGT`
isolates GRU-based history modeling.

---

## Repository structure

```text
.
├── rnn_agt/                     # the package
│   ├── seeds.py                 # four independent RNG streams from one master seed
│   ├── data.py                  # data-generating mechanisms, censoring, tau calibration
│   ├── models.py                # AFTWRS, NNAFT, RNNAGT
│   ├── losses.py                # Gehan-WRS objective, exact and subsampled
│   ├── sampling.py              # pair sampler with explicit inclusion probabilities
│   ├── metrics.py               # vectorised IPCW C-index and AMSE
│   ├── train.py                 # one trainer, all three model classes
│   ├── evaluation.py            # repeated splits, k-fold CV, paired differences
│   ├── cox_bridge.py            # R/Python bridge for the Cox comparators
│   ├── latex.py                 # emits table rows for the manuscript
│   └── diagnostics.py           # correctness checks
│
├── experiments/                 # command-line drivers, one per table
│   ├── run_diagnostics.py       # run this first
│   ├── run_simulation_tables.py         Tables 1-3
│   ├── run_dependence_robustness.py     Table 5
│   ├── run_benchmark_rows.py            Table 7
│   ├── run_ablation_and_splits.py       Tables 6 and 7
│   └── run_capacity_sweep.py            Table 8
│
├── simulation/
│   ├── model_demo.ipynb                 walkthrough of the three model classes
│   ├── loss_sensitivity.ipynb           sensitivity to loss and censoring choices
│   ├── subsampling_sensitivity.ipynb    Table 4
│   ├── high_dimensional.ipynb           behaviour as the covariate dimension grows
│   └── figures.ipynb                    manuscript figures
│
├── application/
│   ├── real_data_analysis.ipynb         Tables 6-9
│   ├── split_sensitivity.ipynb          variability across random train/test splits
│   ├── Dataset1_Data_Preprocessing.R            colorectal readmission -> data/crc.csv
│   ├── Dataset1_Classical_Recurrent_Event_Models.R
│   ├── Dataset2_Data_Preprocessing.R            CGD -> data/cgd.csv
│   └── Dataset2_Classical_Recurrent_Event_Models.R
│
├── literature_review/           # background reading
├── data/                        # generated CSVs (git-ignored)
└── results/                     # tables, figures, JSON (git-ignored)
```

All paths are relative to the repository root. Run scripts from there
(`python experiments/...`, `Rscript application/...`); the notebooks add the
root to `sys.path` themselves.

### Datasets

**Dataset 1** is the colorectal cancer readmission data from
`frailtypack::readmission` (403 subjects). **Dataset 2** is the chronic
granulomatous disease data from `survival::cgd` (128 subjects). Both ship with
public R packages, so the analyses reproduce end to end without a data-access
request.

---

## Which script produces which table

| Table | Contents | Produced by |
|---|---|---|
| 1 | Interaction mean function | `experiments/run_simulation_tables.py` |
| 2 | GAM-type nonlinear mean function | same |
| 3 | Linear mean function | same |
| 4 | Sub-sampling sensitivity | `simulation/subsampling_sensitivity.ipynb` |
| 5 | Dependence-mechanism robustness | `experiments/run_dependence_robustness.py` |
| 6 | Ablation ladder | `experiments/run_ablation_and_splits.py` |
| 7 | Repeated splits and 5-fold CV | same |
| 8 | Capacity sweep | `experiments/run_capacity_sweep.py` |

Every driver writes a `.json` holding the raw per-replicate numbers and a `.tex`
fragment ready to paste into the manuscript.

Every emitter's column count is checked against the manuscript: 9 for
Tables 1-3, 6 for Table 4, 4 for Table 5, 7 for Table 6, 7 for Table 7, 7 for
Table 8. Tables 1-4 additionally use
`\multirow` spans in their label columns, so the emitters produce those rather
than flat shaded rows.

Tables 1-4 report **training duration as a column**. The epoch grid differs by
sample size, since a larger training set reaches the same effective number of
gradient steps in fewer passes: `{5, 10, 15}` at `n=1000` and `{2, 3, 5}` at
`n=5000` for Tables 1-3, `{5, 10}` and `{2, 3}` for Table 4. These are set in
`EPOCH_GRID` in `run_simulation_tables.py` and can be overridden with
`--epochs 5 10 15`.

A single fit produces every epoch column, via `TrainConfig(eval_at_epochs=...)`.
The optimizer state at epoch 5 does not depend on whether training later
continues to 15, so checkpointing is exactly equivalent to refitting to each
epoch count with the same seed, at a third of the cost. The emitted `.tex`
fragment carries a comment giving the column order.

---

## Which script produces which figure

| Figure | Contents | Produced by | Output files |
|---|---|---|---|
| 1 | Unrolled stacked GRU architecture | TikZ in the manuscript | none |
| 2 | Training loss, all mean functions x error distributions (9 panels) | `simulation/figures.ipynb` | `loss_{normal,gumbel,log}_nonlinear.png`, `..._n1000_gam.png`, `..._n1000_linear.png` |
| 3 | High-dimensional sweep, linear DGP (4 panels) | `simulation/high_dimensional.ipynb` | `amse_c-index_plots.png`, `train_test_cindex.png`, `train_test_amse.png`, `loss_trajectorie.png` |
| 4 | High-dimensional sweep, nonlinear DGP (4 panels) | same notebook | the same four names with a `1` suffix |
| 5 | NN-AFT architecture | TikZ in the manuscript | none |

Figures 1 and 5 are drawn in LaTeX and have no code file; edit them in the
manuscript source. The other three read their panels from `manuscript/images/`.

Both notebooks write to `manuscript/images/` when that directory exists and to
`results/` otherwise. **The filenames are fixed by the `\includegraphics` calls
in the manuscript and must not be changed.** Note that `log` in the Figure 2
filenames abbreviates the *logistic* error distribution, not a log transform,
and that the unsuffixed Figure 3 names are easy to collide with -- nothing else
in the repository writes them.

Two additional plots are produced for inspection rather than for the
manuscript: `simulation/subsampling_sensitivity.ipynb` writes
`results/subsampling_sensitivity.png`, and
`application/split_sensitivity.ipynb` draws the distribution of the C-index
across repeated splits inline.

---

## Installation

### Python

```bash
git clone https://github.com/EmmanuelMasavoDjegou/Modeling-Recurrent-Events-Using-Recurrent-Neural-Networks-in-AGTs.git
cd Modeling-Recurrent-Events-Using-Recurrent-Neural-Networks-in-AGTs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10 or newer.

### R

R 4.3 or newer is recommended; Debian 12 ships 4.2.2 in main, which is older
than current CRAN sources expect. Add the
[CRAN repository](https://cran.r-project.org/bin/linux/debian/) to upgrade.

The R scripts install their own R packages on first run, but a few system
libraries must be present first:

```bash
sudo apt install -y cmake libuv1-dev libcurl4-openssl-dev libssl-dev \
                    libxml2-dev libfontconfig1-dev libfreetype6-dev \
                    libharfbuzz-dev libfribidi-dev libpng-dev \
                    libtiff-dev libjpeg-dev
```

`cmake` is needed only if you install packages outside the required set: some
CRAN packages build their bundled C libraries with it, and its absence shows up
as `CMAKE NOT FOUND` partway through a configure step.

`libuv1-dev` is the one most often missed. Without it `fs` fails to configure
with `uv.h: No such file or directory`, which cascades through `sass`, `bslib`
and `shiny` to `frailtypack`. Since `install.packages()` only warns on build
failure, this can look like success until a `library()` call fails much later.

---

## Verify the installation

```bash
python experiments/run_diagnostics.py
```

Five checks, all of which should pass:

```
[PASS] vectorised C-index matches reference loop
[PASS] censoring reaches the outcome (no latent leak)
[PASS] subsampled Gehan-WRS is unbiased (weighted)
[PASS] subsampled Gehan-WRS is unbiased (unweighted)
[PASS] predictor is history-predictable (A3)
```

The first confirms the fast concordance implementation agrees with a direct
transcription of the definition. The third is a Monte Carlo test that the
subsampled objective is unbiased for the full one, which is what licenses the
stochastic subgradient argument. The last confirms no predictor can see its own
outcome. Run this before trusting any number the package produces.

---

## Reproducing the results

### Simulations

```bash
python experiments/run_simulation_tables.py --replicates 500      # Tables 1-3
python experiments/run_dependence_robustness.py --replicates 500  # Table 5
```

Add `--quick` to either for a two-replicate pipeline test.

Progress lines report both metrics, since they answer different questions and
can move in opposite directions:

```
[  1/108] linear  normal  frailty  cens=0.25 n=1000  E5:0.939/30.91  E10:0.939/50.41
          E15:0.938/51.50  (achieved cens 0.27, shift -4.12, eta 415 min)
```

The format is `E<epoch>:<C-index>/<AMSE>`. `achieved cens` confirms the
calibrated `tau` hit its target, and `shift` is the fitted intercept applied
before AMSE. A shift growing with the epoch index is expected -- the level is
unidentified by the objective and drifts with gradient steps -- but a large one
is worth noting alongside the results. The full grid for
Tables 1-3 is 108 cells times the replicate count, which is a cluster job rather
than a laptop one; `--mean-funcs interaction` runs one table at a time.

### Real data

Four stages, because the Cox comparators are fitted in R and must see the same
splits as the neural models.

```bash
# 1. preprocess -> data/crc.csv, data/cgd.csv
Rscript application/Dataset1_Data_Preprocessing.R
Rscript application/Dataset2_Data_Preprocessing.R

# 2. fit the AFT models and write the split assignments
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd.csv --crc data/crc.csv \
    --splits 200 --splits-out results/splits

# 3. fit the Cox models on those same splits, exporting held-out
#    linear predictors
Rscript application/Dataset1_Classical_Recurrent_Event_Models.R \
    --data data/crc.csv --splits results/splits/splits_crc.csv \
    --out results/cox_lp_crc.csv
Rscript application/Dataset2_Classical_Recurrent_Event_Models.R \
    --data data/cgd.csv --splits results/splits/splits_cgd.csv \
    --out results/cox_lp_cgd.csv

# 4. re-run stage 2 with the Cox predictors merged in
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd.csv --crc data/crc.csv \
    --splits 200 --splits-out results/splits \
    --cox-lp-cgd results/cox_lp_cgd.csv \
    --cox-lp-crc results/cox_lp_crc.csv

# Table 8
python experiments/run_capacity_sweep.py \
    --cgd data/cgd.csv --crc data/crc.csv --splits 200
```

The R scripts export **linear predictors**, not concordance indices, and Python
scores them with the same IPCW estimator used for the neural models. Keeping a
single concordance implementation in the project is deliberate: an in-sample or
differently-weighted C-index computed on the R side would not be comparable with
the out-of-sample IPCW values reported for the AFT models.

---

## Notes on use

**AMSE is reported with the intercept fixed.** The Gehan objective is invariant
to a location shift of the predictor: adding a constant to every prediction
leaves all pairwise residual differences unchanged, so training carries no
information about the level and the intercept is not identified. Concordance is
unaffected, but AMSE is an absolute-error criterion and is not: an
uncalibrated level displaced by `c` inflates AMSE by roughly `c**2`, and since
the displacement grows with gradient steps, an uncalibrated AMSE gets *worse*
with more training. `metrics.estimate_location_shift` fits the shift from the
IPCW-weighted mean residual on the **training** partition and applies it to
both, so no test outcome fits it. It is on by default
(`TrainConfig(calibrate_location=True)`) and the fitted value is recorded as
`location_shift`. `run_diagnostics.py` demonstrates the invariance directly.

**Data format.** The drivers expect one row per gap, ordered within subject,
with columns `id`, `gap_time`, `delta` and the covariates. `gap_time` must be
strictly positive, since the models work on the log scale; the preprocessing
scripts raise rather than filtering, because dropping rows on one side only
would put the Cox and AFT models on different data and break the pairing that
Table 7 depends on. See `data/README.md`.

**Sign convention for the Cox bridge.** A Cox linear predictor is on the
log-hazard scale, where larger means shorter gaps; the AFT models predict log
gap time, where larger means longer. Python negates the linear predictor once,
in `cox_bridge.score_cox_predictions`. Never negate it in R as well.
`check_orientation` warns if the mean Cox C-index falls below 0.45, which is
what a double negation looks like.

**AFT-WRS needs a larger learning rate.** With three parameters against the
GRU's tens of thousands, it barely moves at `3e-4`; the drivers default it to
`1e-2` via `--lr-linear`. An undertrained comparator would flatter RNN-AGT for
the wrong reason, so confirm it has converged before reporting it. On a linear
data-generating process with known coefficients it recovers the correct signs
and ratio.

**NN-AFT is parameter-matched to the GRU** via `models.matched_mlp_width`, so a
gap between those two rungs cannot be attributed to one model simply being
larger.

**Nothing is reported from a single train-test split.** Every real-data
quantity is averaged over `B=200` repeated stratified partitions, with all
methods fitted on identical splits so that comparisons are paired. Table 7
reports training and test columns side by side: they use the same estimator but
on partitions with different censoring distributions, so the IPCW weights are
on different scales and the two are not two draws of one quantity.

**Win rate accompanies every paired difference.** A method can carry a positive
mean increment while losing on 40% of splits; those are different claims, and
both are reported.

**Reproducibility.** Data generation, network initialization, pair subsampling
and the train/test partitioner each draw from a separately seeded generator
derived from one master seed, so any one component can be varied while the
others are held fixed. Nothing in the package uses the global NumPy RNG.

**GPU.** The models are small and the bottleneck is pair sampling and the IPCW
metrics, both CPU-bound. A GPU gives little benefit and can be slower.
`--device cuda` is available if you want to measure it.

---

## Citation

See `CITATION.cff`. Licensed under MIT.
