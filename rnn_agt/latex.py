"""
Emit LaTeX fragments that drop straight into the manuscript's ``\\PH{}`` slots.

The manuscript carries 47 placeholders across Tables 4, 6, 7, 8 and 9.
Transcribing that many numbers by hand is where transcription errors come
from, so every driver in ``experiments/`` writes its table with these helpers
and you paste the block over the corresponding ``\\PH{...}`` cells.

Formatting conventions follow the existing tables: concordance to three
decimals, AMSE to two, standard deviations in parentheses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def fmt(value: float, digits: int = 3) -> str:
    """Format a number, rendering NaN as an em-free placeholder marker."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\PH{n/a}"
    return f"{value:.{digits}f}"


def fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    """``0.637 (0.041)``."""
    if mean is None or np.isnan(mean):
        return r"\PH{n/a}"
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def fmt_c_amse(cindex: float, amse_val: float) -> str:
    """``0.637 / 8.42``, matching the existing simulation tables."""
    if np.isnan(cindex) and np.isnan(amse_val):
        return r"\PH{n/a}"
    return f"{fmt(cindex, 3)} / {fmt(amse_val, 2)}"


def fmt_diff_ci(d: Dict[str, float], digits: int = 3) -> str:
    """``+0.021 (95\\% CI: 0.004, 0.038)``."""
    if np.isnan(d.get("mean", np.nan)):
        return r"\PH{n/a}"
    sign = "+" if d["mean"] >= 0 else ""
    return (
        f"{sign}{d['mean']:.{digits}f} "
        f"(95\\% CI: {d['lo']:.{digits}f}, {d['hi']:.{digits}f})"
    )


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------


def table_rows(
    rows: List[Sequence[str]],
    row_colors: Optional[Sequence[str]] = None,
    bold_last: bool = False,
) -> str:
    """Assemble ``&``-separated rows with optional alternating cell colours.

    ``row_colors`` cycles the ``rowA``/``rowB`` colours already defined in the
    manuscript preamble, so emitted rows match the surrounding table.
    """
    out = []
    for i, cells in enumerate(rows):
        cells = list(cells)
        if bold_last and i == len(rows) - 1:
            cells[0] = r"\textbf{" + cells[0] + "}"
        if row_colors:
            color = row_colors[i % len(row_colors)]
            cells = [rf"\cellcolor{{{color}}}{c}" for c in cells]
        out.append("  " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def dependence_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    mechanisms: Sequence[str],
    mechanism_labels: Dict[str, str],
    models: Sequence[str] = ("agt_wrs", "nn_agt", "rnn_agt"),
) -> str:
    """Table 5: dependence-mechanism robustness.

    ``results[mechanism][model]`` supplies ``test_cindex`` and ``test_amse``.
    """
    rows = []
    for mech in mechanisms:
        cells = [mechanism_labels.get(mech, mech)]
        for m in models:
            r = results.get(mech, {}).get(m, {})
            cells.append(
                fmt_c_amse(r.get("test_cindex", np.nan), r.get("test_amse", np.nan))
            )
        rows.append(cells)
    return table_rows(rows, row_colors=("rowA", "rowB"))


def ablation_table(
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    deltas: Dict[str, Dict[str, Dict[str, float]]],
    datasets: Sequence[str] = ("cgd", "crc"),
) -> str:
    """Table 6: the ablation ladder with delta increments.

    ``summaries[dataset][model]`` holds mean/sd for each metric;
    ``deltas[dataset][contrast]`` holds paired differences.
    """
    spec = [
        ("AGT-WRS", "agt_wrs", "--", "--"),
        ("NN-AGT", "nn_agt", r"\checkmark", "--"),
        ("RNN-AGT", "rnn_agt", r"\checkmark", r"\checkmark"),
    ]
    rows = []
    for label, key, nonlin, hist in spec:
        cells = [label, nonlin, hist]
        for ds in datasets:
            c = summaries.get(ds, {}).get(f"{key}:cindex", {})
            a = summaries.get(ds, {}).get(f"{key}:amse", {})
            cells.append(fmt_mean_sd(c.get("mean", np.nan), c.get("sd", np.nan), 3))
            cells.append(fmt_mean_sd(a.get("mean", np.nan), a.get("sd", np.nan), 2))
        rows.append(cells)

    body = table_rows(rows, row_colors=("rowA", "rowB"), bold_last=True)

    delta_rows = []
    for contrast, label in (
        ("nonlinearity", r"$\Delta$ nonlinearity (NN-AGT $-$ AGT-WRS)"),
        ("history", r"$\Delta$ history (RNN-AGT $-$ NN-AGT)"),
    ):
        cells = [rf"\multicolumn{{3}}{{l}}{{{label}}}"]
        for ds in datasets:
            d_c = deltas.get(ds, {}).get(f"{contrast}:cindex", {})
            d_a = deltas.get(ds, {}).get(f"{contrast}:amse", {})
            cells.append(fmt(d_c.get("mean", np.nan), 3))
            cells.append(fmt(d_a.get("mean", np.nan), 2))
        delta_rows.append("  " + " & ".join(cells) + r" \\")

    return body + "\n  \\midrule[0.6pt]\n" + "\n".join(delta_rows)


def repeated_splits_table(
    split_summary: Dict[str, Dict[str, Dict[str, float]]],
    train_summary: Dict[str, Dict[str, Dict[str, float]]],
    cv_summary: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
    model_labels: Dict[str, str],
    datasets: Sequence[str] = ("cgd", "crc"),
) -> str:
    """Body rows for Table 7: train, test and cross-validated C-index.

    Training and test columns are both reported. They use the same estimator
    but are computed on partitions with different censoring distributions, so
    the IPCW weights are on different scales and the two are not two draws of
    one quantity. Averaging over the splits removes the split-to-split noise
    that makes any single training value hard to interpret.
    """
    rows = []
    for m in models:
        cells = [model_labels.get(m, m)]
        for ds in datasets:
            tr = train_summary.get(ds, {}).get(m, {})
            cells.append(fmt_mean_sd(tr.get("mean", np.nan), tr.get("sd", np.nan), 3))
            te = split_summary.get(ds, {}).get(m, {})
            cells.append(fmt_mean_sd(te.get("mean", np.nan), te.get("sd", np.nan), 3))
            cv = cv_summary.get(ds, {}).get(m, {})
            cells.append(fmt(cv.get("test_cindex", np.nan), 3))
        rows.append(cells)
    return table_rows(rows, row_colors=("rowA", "rowB"), bold_last=True)


def capacity_table(
    results: Dict[tuple, Dict[str, Dict[str, float]]],
    layers: Sequence[int],
    dims: Sequence[int],
    param_counts: Dict[tuple, int],
    datasets: Sequence[str] = ("cgd", "crc"),
) -> str:
    """Table 8: capacity sweep.

    ``results[(L, d)][dataset]`` holds mean/sd for cindex and amse.
    """
    rows = []
    for L in layers:
        for d in dims:
            cells = [str(L), str(d), f"{param_counts.get((L, d), 0):,}"]
            for ds in datasets:
                r = results.get((L, d), {}).get(ds, {})
                cells.append(
                    fmt_mean_sd(r.get("cindex_mean", np.nan), r.get("cindex_sd", np.nan), 3)
                )
                cells.append(
                    fmt_mean_sd(r.get("amse_mean", np.nan), r.get("amse_sd", np.nan), 2)
                )
            rows.append(cells)
    return table_rows(rows, row_colors=("rowA", "rowB"))


def write_fragment(path: str, title: str, body: str) -> None:
    """Write a table fragment with a header saying where it belongs."""
    with open(path, "w") as fh:
        fh.write(f"%% {title}\n")
        fh.write("%% Generated by the rnn_agt experiment drivers.\n")
        fh.write("%% Paste over the corresponding \\PH{} cells in the manuscript.\n\n")
        fh.write(body)
        fh.write("\n")


# --------------------------------------------------------------------------
# Tables 1-3: mean-function grids
# --------------------------------------------------------------------------

ERROR_LABELS = {"normal": "Normal", "gumbel": "Gumbel", "logistic": "Logistic"}
ERROR_COLORS = {"normal": "errNormal", "gumbel": "errGumbel",
                "logistic": "errLogistic"}
DEPENDENCE_SHORT = {"frailty": "Frailty", "ar1": "AR(1)", "nar1": "NAR(1)",
                    "ar2": "AR(2)", "event_dependent": "Event-dep."}


def simulation_table(
    acc: Dict[tuple, Sequence],
    mean_func: str,
    errors: Sequence[str],
    dependence: Sequence[str],
    censoring: Sequence[float],
    n_trains: Sequence[int],
    epochs_for,
) -> str:
    """Body rows for Tables 1-3, matching the manuscript's multirow layout.

    The first two columns are ``\\multirow`` spans -- the error label over all
    its dependence blocks, and the dependence label over its censoring levels --
    with the error cell carrying its own background colour. Only the data
    columns take the alternating ``rowA``/``rowB`` shading, and a
    ``\\cmidrule`` separates the dependence blocks. Emitting flat coloured rows
    instead would shade the label columns and lose the grouping.

    ``acc`` is keyed by ``(mean_func, error, dependence, censoring, n_train,
    epoch)`` and holds a sequence of ``(cindex, amse)`` pairs.
    """
    n_dep = len(dependence)
    n_cens = len(censoring)
    n_data_cols = sum(len(epochs_for(n)) for n in n_trains)
    last_col = 3 + n_data_cols

    out = []
    shade = 0
    for e_i, err in enumerate(errors):
        for d_i, dep in enumerate(dependence):
            for c_i, cens in enumerate(censoring):
                cells = []
                # column 1: error label, spanning every row for this error
                if d_i == 0 and c_i == 0:
                    cells.append(
                        rf"\multirow{{{n_dep * n_cens}}}{{*}}"
                        rf"{{\cellcolor{{{ERROR_COLORS.get(err, 'errNormal')}}}"
                        rf"{ERROR_LABELS.get(err, err)}}}"
                    )
                else:
                    cells.append("")
                # column 2: dependence label, spanning its censoring levels
                if c_i == 0:
                    cells.append(
                        rf"\multirow{{{n_cens}}}{{*}}"
                        rf"{{{DEPENDENCE_SHORT.get(dep, dep)}}}"
                    )
                else:
                    cells.append("")
                # column 3: censoring percentage, unshaded
                cells.append(f"{int(round(cens * 100))}")

                color = "rowA" if shade % 2 == 0 else "rowB"
                shade += 1
                for n_tr in n_trains:
                    for ep in epochs_for(n_tr):
                        key = (mean_func, err, dep, cens, n_tr, ep)
                        vals = acc.get(key)
                        if vals:
                            arr = np.asarray(vals, dtype=float)
                            txt = fmt_c_amse(
                                float(np.nanmean(arr[:, 0])),
                                float(np.nanmean(arr[:, 1])),
                            )
                        else:
                            txt = r"\PH{n/a}"
                        cells.append(rf"\cellcolor{{{color}}}{txt}")

                out.append("  " + " & ".join(cells) + r" \\")

            if d_i < n_dep - 1:
                out.append(rf"  \cmidrule(lr){{2-{last_col}}}")
        if e_i < len(errors) - 1:
            out.append(r"  \midrule[0.6pt]")
    return "\n".join(out)


def subsampling_table(
    results: Dict[tuple, Dict[str, float]],
    s_grid: Sequence[int],
    b_grid: Sequence[int],
    n_epochs: Dict[int, Sequence[int]],
) -> str:
    """Body rows for Table 4, with ``s`` as a multirow span over its ``b`` values.

    ``results`` is keyed by ``(s, b, n_train, epoch)`` and holds a dict with
    ``cindex`` and ``amse``.
    """
    out = []
    shade = 0
    for s_i, s_val in enumerate(s_grid):
        for b_i, b_val in enumerate(b_grid):
            cells = []
            if b_i == 0:
                cells.append(rf"\multirow{{{len(b_grid)}}}{{*}}{{{s_val}}}")
            else:
                cells.append("")
            color = "rowA" if shade % 2 == 0 else "rowB"
            shade += 1
            cells.append(rf"\cellcolor{{{color}}}{b_val}")
            for n_tr, eps in n_epochs.items():
                for ep in eps:
                    r = results.get((s_val, b_val, n_tr, ep), {})
                    txt = fmt_c_amse(r.get("cindex", np.nan), r.get("amse", np.nan))
                    cells.append(rf"\cellcolor{{{color}}}{txt}")
            out.append("  " + " & ".join(cells) + r" \\")
    return "\n".join(out)

