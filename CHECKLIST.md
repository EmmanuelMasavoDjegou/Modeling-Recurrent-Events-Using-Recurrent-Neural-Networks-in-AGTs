# Deliverables checklist

Every table, figure and unfilled placeholder in the manuscript, with the code
that produces it. Tick items off as you go.

Regenerate the placeholder counts at any time with:

```bash
grep -c 'PH{' path/to/Optimal-Design-layout.tex
```

Counts below are as of the current draft: **143 placeholders**, 124 inside
tables and 19 in prose.

---

## Tables

Run everything from the repository root.

| # | Label | Contents | Command | Slots | Done |
|---|---|---|---|---|---|
| 1 | `tab:nonlinear_interactions` | Interaction mean function | `run_simulation_tables.py --mean-funcs interaction` | 0 | ☑ **filled**, SE 0.0005 |
| 2 | `tab:GAM-type-Nonlinear` | GAM-type nonlinear | `run_simulation_tables.py --mean-funcs gam` | 0 | ☑ **filled**, SE 0.0006 |
| 3 | `tab:Linear-mean` | Linear mean function | `run_simulation_tables.py --mean-funcs linear` | 0 | ☑ **filled**, SE 0.0013 |
| 4 | `tab:subsampling_sensitivity` | Sub-sampling sensitivity | `run_subsampling_sensitivity.py` | 0 | ☑ **filled**, SE 0.0009 |
| 5 | `tab:depend_robust` | Dependence-mechanism robustness | `run_dependence_robustness.py` | 0 | ☑ **filled**, SE 0.0010 |
| 6 | `tab:ablation` | Ablation ladder | `run_ablation_and_splits.py` | 21 | ☐ |
| 7 | `tab:repeated_splits` | Repeated splits + 5-fold CV | `run_ablation_and_splits.py` | 38 | ☐ |
| 8 | `tab:capacity` | Capacity sweep | `run_capacity_sweep.py` | 40 | ☐ |

**Tables 1–3 have only 2 slots each** because the body is generated wholesale by
`latex.simulation_table` — paste the emitted `.tex` fragment over the body rows.
The 2 remaining slots are the standard-error footnote (C-index and AMSE).

**Tables 6 and 7 come from one command.** Do not run it twice; both fragments
are written by the same invocation, and the ablation increments in Table 6 are
paired differences computed on exactly the splits summarised in Table 7.

### Order to run them

1. **Simulations** (no data dependency):
   `run_simulation_tables.py`, `run_dependence_robustness.py`,
   `run_subsampling_sensitivity.py`
2. **R preprocessing** → `data/crc.csv`, `data/cgd.csv`
3. **`run_ablation_and_splits.py`** → writes `results/splits/`
4. **R Cox scripts** on those splits → `results/cox_lp_*.csv`
5. **`run_ablation_and_splits.py` again** with `--cox-lp-*` → fills the Cox rows
   of Table 7
6. **`run_capacity_sweep.py`**

Step 5 is easy to forget. Without it the WLW, PWP-TT and PWP-GT rows of Table 7
stay blank.

---

## Figures

| # | Label | Contents | Produced by | Done |
|---|---|---|---|---|
| 1 | `fig:deep_gru` | Unrolled stacked GRU architecture | TikZ in the manuscript | n/a |
| 2 | `fig:loss_all_layout` | Training loss, 9 panels | `simulation/figures.ipynb` | ☐ |
| 3 | `fig:highdim_linear` | High-dim sweep, linear DGP | `simulation/high_dimensional.ipynb` | ☐ |
| 4 | `fig:highdim_nonlinear` | High-dim sweep, nonlinear DGP | same notebook | ☐ |
| 5 | `fig:nn_agt` | NN-AGT architecture | TikZ in the manuscript | n/a |

Figures 1 and 5 need no code; edit them in the manuscript source.

### The 17 image files

Both notebooks write to `manuscript/images/` if it exists, otherwise
`results/`. **Filenames are fixed by the `\includegraphics` calls and must not
be changed.**

Figure 2 — `simulation/figures.ipynb`:

```
loss_normal_nonlinear.png      loss_gumbel_nonlinear.png      loss_log_nonlinear.png
loss_normal_n1000_gam.png      loss_gumbel_n1000_gam.png      loss_log_n1000_gam.png
loss_normal_n1000_linear.png   loss_gumbel_n1000_linear.png   loss_log_n1000_linear.png
```

Figure 3 — `simulation/high_dimensional.ipynb`, linear DGP:

```
amse_c-index_plots.png   train_test_cindex.png   train_test_amse.png   loss_trajectorie.png
```

Figure 4 — same notebook, nonlinear DGP, `1` suffix:

```
amse_c-index_plots1.png  train_test_cindex1.png  train_test_amse1.png  loss_trajectorie1.png
```

`log` abbreviates the **logistic** error distribution, not a log transform. The
unsuffixed Figure 3 names are collision-prone; nothing else in the repository
writes them.

---

## Prose placeholders (19)

These are not filled by any script.

### Software details — Section 5.3 (9 slots)

| Line | Needs |
|---|---|
| 2020 | Python version |
| 2021 | PyTorch version |
| 2025 | R version |
| 2025 | `survival` version |
| 2026 | AGT-WRS implementation and version |
| 2027 | CPU/GPU model, cores, RAM |
| 2028 | Runtime, full simulation suite |
| 2029 | Runtime, two data analyses |
| 2038 | Zenodo DOI |

Get the versions with:

```bash
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__)"
R -e 'cat(R.version.string, as.character(packageVersion("survival")))'
nproc; free -g; lscpu | grep "Model name"
```

The Zenodo DOI comes from creating a GitHub release and connecting the
repository to Zenodo. A bare GitHub link can be force-pushed or renamed, which
is why the archived version is cited alongside it.

### Abstract (2 slots, lines 180 and 215)

Both say *"summarise once the repeated-split results are available"*. The
abstract and trans-abstract must match — **edit both**.

Write this only after Table 7 is populated. It replaces the "0.02 concordance
units" claim, which came from the single split and no longer appears anywhere.

### Interpretation paragraphs (6 slots)

| Line | Section | Depends on |
|---|---|---|
| 1948 | §4.3.5 dependence robustness | Table 5 |
| 1960 | §4.3.5, which pattern was observed | Table 5 |
| 2071 | §5.4 benchmarking | Table 7 |
| 2338 | §5.5 ablation | Table 6 |
| 2404 | §5.6 repeated splits | Table 7 |
| 2464 | §5.7 capacity | Table 8 |

**Read these before filling them.** They commit you in advance to reporting
unfavourable findings: a null or negative Δ-history, an advantage that narrows
once dependence stops being linear, smaller networks winning the capacity
sweep. Stating that standard in advance is what keeps the reported comparison
honest, but it is a real commitment. Soften them before submission if you are
not prepared to honour it.

### Lines 47 and 49

Empty `\PH{}` in the preamble macro definition, not content. Ignore.

---

## Standard-error footnotes

Every simulation table carries a footnote with the maximum Monte Carlo standard
error across its own cells, for C-index and AMSE separately. Each driver prints
both at the end of its run:

```
max Monte Carlo SE  C-index: 0.0041   AMSE: 0.0730
```

The footnotes report the C-index standard error only, which is what Section 4.1
justifies. The drivers also persist every per-replicate value under `"raw"` in
the JSON, so any statistic not computed at run time can be derived later without
repeating the grid.

That number fills the table's `\PH{max SE}` slot. Tables 1-3 are already
filled: 0.0005, 0.0006 and 0.0013 respectively. Section 4.1 promises
these, so a blank footnote is a visible broken promise.

---

## Before submitting

- [ ] All 143 placeholders filled — verify with `grep -c 'PH{'`
- [ ] Both abstracts updated and identical
- [ ] Interpretation paragraphs read and agreed with
- [ ] Real `USG.cls` dropped into `classes/`
- [ ] All 17 figure images in `manuscript/images/`
- [ ] `\revfinaltrue` set in the preamble for the clean copy
- [ ] Recompile: any missed placeholder prints a loud
      `??PLACEHOLDER NOT FILLED??`
- [ ] Repository pushed, release tagged, Zenodo DOI minted
- [ ] Response letter numbers cross-checked against the final PDF

Setting `\revfinaltrue` is the safety net: it turns silent gaps into visible
errors, so run it before the final read-through rather than after.
