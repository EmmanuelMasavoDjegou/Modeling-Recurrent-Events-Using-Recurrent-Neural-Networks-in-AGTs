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
    models: Sequence[str] = ("aft_wrs", "nn_aft", "rnn_agt"),
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
    """Table 8: the ablation ladder with delta increments.

    ``summaries[dataset][model]`` holds mean/sd for each metric;
    ``deltas[dataset][contrast]`` holds paired differences.
    """
    spec = [
        ("AFT-WRS", "aft_wrs", "--", "--"),
        ("NN-AFT", "nn_aft", r"\checkmark", "--"),
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
        ("nonlinearity", r"$\Delta$ nonlinearity (NN-AFT $-$ AFT-WRS)"),
        ("history", r"$\Delta$ history (RNN-AGT $-$ NN-AFT)"),
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
    cv_summary: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
    model_labels: Dict[str, str],
    datasets: Sequence[str] = ("cgd", "crc"),
) -> str:
    """Table 9: repeated splits and cross-validation."""
    rows = []
    for m in models:
        cells = [model_labels.get(m, m)]
        for ds in datasets:
            s = split_summary.get(ds, {}).get(m, {})
            cells.append(fmt_mean_sd(s.get("mean", np.nan), s.get("sd", np.nan), 3))
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
    """Table 10: capacity sweep.

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


def benchmark_rows(results: Dict[str, Dict[str, float]]) -> str:
    """The two rows added to Table 7 (AFT-WRS and NN-AFT, single split)."""
    rows = [
        [r"\rev{AFT-WRS (linear gap-time)}",
         fmt(results.get("aft_wrs", {}).get("cgd", np.nan), 3),
         fmt(results.get("aft_wrs", {}).get("crc", np.nan), 3)],
        [r"\rev{NN-AFT (nonlinear, no history)}",
         fmt(results.get("nn_aft", {}).get("cgd", np.nan), 3),
         fmt(results.get("nn_aft", {}).get("crc", np.nan), 3)],
    ]
    return table_rows(rows)


def write_fragment(path: str, title: str, body: str) -> None:
    """Write a table fragment with a header saying where it belongs."""
    with open(path, "w") as fh:
        fh.write(f"%% {title}\n")
        fh.write("%% Generated by the rnn_agt experiment drivers.\n")
        fh.write("%% Paste over the corresponding \\PH{} cells in the manuscript.\n\n")
        fh.write(body)
        fh.write("\n")
