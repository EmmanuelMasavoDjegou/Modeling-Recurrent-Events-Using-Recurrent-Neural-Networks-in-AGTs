<!-- Corrections applied in the v2.0.0 release. Referenced from README.md. -->

# Corrections relative to the original submission

Two of these affect published numbers. Please check them before re-running
anything, because if they are confirmed, the simulation results in the
submitted manuscript need regenerating rather than merely extending.

### 0. The Gehan hinge sign was inverted (most consequential)

As written in the manuscript and implemented in every notebook, the penalty is
`max(0, e_anchor - e_compare)`. The Gehan objective whose subgradient
reproduces the Gehan estimating function

```
U(b) = sum_i sum_j delta_i (Z_i - Z_j) 1{e_i <= e_j}
```

is `sum delta_i (e_i - e_j)^-` with `a^- = |a| 1(a<0)`, which expands to
`max(0, e_compare - e_anchor)`. Differentiating confirms it: the active term
`e_j - e_i` has gradient `Z_i - Z_j`, which is exactly `U(b)`. The two
orientations push the predictor in opposite directions.

Measured effect, interaction setting with AR(1) dependence, test C-index
against an oracle that predicts the true conditional mean:

| Censoring | Original sign | Corrected | Oracle |
|---|---|---|---|
| 25% | 0.882 | **0.940** | 0.940 |
| 65% | 0.621 | **0.916** | 0.963 |

The corrected objective essentially attains oracle discrimination. The original
loses a third of the gap at heavy censoring.

**This defect and the next one partially cancelled** — training against
uncensored outcomes compensated for a loss pushing the wrong way — which is
almost certainly why neither was caught. They look tolerable at light censoring
and only come apart at heavy censoring.

**The manuscript equation needs correcting too**, not just the code. Equation
(3.x) for the linear loss and its nonlinear counterpart both write
`[e_lk - e_ij]^-` with `[a]^- = max(0,-a)`, which yields the inverted form.

### 1. Censoring never reached the outcome

`apply_censoring` wrote truncated gaps to `subj['censored_gaps']` and indicators
to `subj['delta']`. But `prepare_subjects_for_nn` then handed the model
`subj['log_gaps']` — the *latent, uncensored* gap times:

```python
# original
def prepare_subjects_for_nn(subject_list):
    return [{'covariates': s['covariates'],
             'log_gaps':   s['log_gaps'],      # <- latent, not censored
             'delta':      s['delta']} for s in subject_list]
```

So the network was trained against the truth while being told, via `delta`, that
some records were censored. Residuals for censored records used a value the
analyst would never observe. Worse, records past the censoring point were kept
as zero-length slots with `delta=0` while carrying their true log gap times, so
padding entered the loss as data.

This is the code-level version of exactly what Reviewer 1 raised in comment 4:
the conflation of `Y_ij = log T_ij` with `Ytilde_ij = log G_ij`. In this package
the two are separate throughout, and `check_censoring_leak` verifies it.

**Likely effect:** optimistic performance, increasing with the censoring
fraction. The 65% incomplete-follow-up columns will be affected most.

### 2. The WRS normalization was absent from the loss

The manuscript's objective is

```
L(w) = (1/n) sum_i sum_j sum_l sum_k [ delta_ij / (K_i* K_l*) ] [e_lk - e_ij]^-
```

The original loss had no `1/(K_i* K_l*)` factor — it was a plain Gehan rank
loss, rescaled by a scalar `(N_total - 1)/s`. The subject-level normalization is
not decoration: it is the entire mechanism by which the WRS construction handles
induced dependent censoring, equalising each subject's contribution regardless
of how many events they accrued. Without it, a subject contributing six gaps
carries six times the weight of one contributing a single gap, and since
accruing many gaps is itself informative about the error process, the estimating
function is biased.

It is also the property Section 3.4.4 relies on when arguing the WRS
construction survives the move to a nonlinear predictor, so implementation and
manuscript have to agree.

Both forms are available: `TrainConfig(weighted_loss=True)` (default, matches
the manuscript) and `weighted_loss=False` (reproduces the original). Running
both measures the practical size of the discrepancy rather than assuming it.

### 3. `f_gam` docstring contradicted its body

The docstring advertised `x1 + 2*x2**3 + sin(0.9*x3)`; the body computed
`x1 + x2**3 + exp(0.9*x3)`. The body matched the manuscript, so only the
docstring was wrong. Corrected.

---

## Third defect found later: the Cox comparison was not like-for-like

Both `Dataset*_Classical_Recurrent_Event_Models.R` scripts reported
`summary(fit)$concordance` — Harrell's C, computed **in sample, unweighted, on
the full dataset with no train/test split**. RNN-AGT's reported C-index is
**out-of-sample and IPCW-weighted**.

Table 7 therefore compared an in-sample unweighted statistic against an
out-of-sample weighted one. In-sample concordance is optimistic by
construction, most severely at small n, which is a plausible explanation for
PWP-GT's 0.648 on the 128-subject CGD data.

The fix is the R/Python bridge:

```bash
# 1. Python writes split assignments (also fits the three AFT models)
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd_preprocessed.csv --crc data/data_cp.csv \
    --splits 200 --splits-out results/splits

# 2. R fits the Cox models on the SAME splits, exports held-out
#    linear predictors (not concordance indices)
Rscript Dataset1_Classical_Recurrent_Event_Models.R \
    --data data_cp.csv --splits results/splits/splits_crc.csv \
    --out results/cox_lp_crc.csv
Rscript Dataset2_Classical_Recurrent_Event_Models.R \
    --data cgd_preprocessed.csv --splits results/splits/splits_cgd.csv \
    --out results/cox_lp_cgd.csv

# 3. Python scores them with the identical IPCW estimator and merges
python experiments/run_ablation_and_splits.py \
    --cgd data/cgd_preprocessed.csv --crc data/data_cp.csv \
    --splits 200 --splits-out results/splits \
    --cox-lp-cgd results/cox_lp_cgd.csv \
    --cox-lp-crc results/cox_lp_crc.csv
```

Exporting linear predictors rather than a concordance computed in R is the
point: there is exactly one concordance implementation in the project, so the
estimand mismatch cannot recur.

**Sign convention.** A Cox linear predictor is on the log-hazard scale (larger
= shorter gap); the AFT models predict log gap time (larger = longer). Python
negates it once, in `cox_bridge.score_cox_predictions`. Never negate it in R as
well. `check_orientation` warns if the mean Cox C-index lands below 0.45, which
is what a double negation looks like.
