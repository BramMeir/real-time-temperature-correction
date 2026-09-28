"""
Paired comparison of the three linear models on the outage windows of the models comparison
(src/scripts/models_comparison_experiment.py), printed as the two LaTeX tables of the paper: the
selection of outage durations in the results (tab:paired) and every duration in the appendix
(tab:paired_full). The error of a window is its RMSE over the whole outage, and every model
reconstructs exactly the same windows, so the comparison works on the per window differences
(src/evaluation/paired_comparison.py).

python -m src.plotting.plot_paired_outage_comparison [--results_dir output/models_comparison_linear]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.evaluation.paired_comparison import WINDOW, SEED, bootstrap_interval, sign_flip_test, holm

RESULTS_DIR = "output/models_comparison_linear"

# Pairs of models to compare: the two models that extend the neighbour regression against it, and the
# two estimators of the same model against each other
CONTRASTS = [
    ("RegressionSARIMAErrors", "LinearRegression"),
    ("ARIMAX", "LinearRegression"),
    ("RegressionSARIMAErrors", "ARIMAX"),
]

REFERENCE = "LinearRegression"

# Outage durations in hours shown in the results table. The comparison and the Holm adjustment still
# run over every duration in the results, which the appendix table lists in full
TABLE_HORIZONS = [1, 4, 12, 48, 168, 720]

# Names as used in the paper
MODEL_NAMES = {
    "RegressionSARIMAErrors": "RegSARIMA",
    "ARIMAX": "SARIMAX",
    "LinearRegression": "neighbour regression",
}
SHORT_NAMES = {**MODEL_NAMES, "LinearRegression": "neighbour reg."}

BOLD_BELOW = 0.05

NUMBER_WORDS = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def load_results(results_dir=RESULTS_DIR):
    """
    Read the comparison results of the linear models and add the RMSE per window.

    Input
    -----
    results_dir: Directory with the result files (default is RESULTS_DIR)

    Output
    ------
    Returns a DataFrame with one row per model, window and horizon.
    """
    paths = glob.glob(os.path.join(results_dir, "*_8_weeks.csv"))
    df = pd.concat(pd.read_csv(path) for path in paths)
    df = df[df["Model"].isin(MODEL_NAMES)].copy()
    df["RMSE"] = np.sqrt(df["MSE"])

    return df


def compare_outages(df, contrasts=CONTRASTS, seed=SEED):
    """
    Compare every pair of models at every outage duration in the results, on the RMSE per window.

    Input
    -----
    df: DataFrame as returned by load_results
    contrasts: List of (model, reference) tuples to compare (default is CONTRASTS)
    seed: Seed of the random generator (default is SEED)

    Output
    ------
    Returns a DataFrame with one row per contrast and duration, holding the error of the reference,
    the mean difference with its confidence interval, the raw and adjusted p value and the share of
    windows in which the model beats its reference.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for horizon in sorted(df["Horizon"].unique()):
        losses = df[df["Horizon"] == horizon].pivot_table(index=WINDOW, columns="Model", values="RMSE")

        for model, reference in contrasts:
            differences = (losses[model] - losses[reference]).values
            lower, upper = bootstrap_interval(differences, rng)

            rows.append({
                "Horizon": horizon,
                "Model": model,
                "Reference": reference,
                "Reference_RMSE": losses[reference].mean(),
                "Difference": differences.mean(),
                "Lower": lower,
                "Upper": upper,
                "P_value": sign_flip_test(differences, rng),
                "Win_rate": float((differences < 0).mean()),
                "Windows": len(differences),
            })

    result = pd.DataFrame(rows)

    # Every pair is tested once per duration, so each pair is its own family
    result["P_holm"] = result.groupby(["Model", "Reference"])["P_value"].transform(
        lambda values: holm(values.values)
    )

    return result


def duration_label(hours):
    """Outage duration as written in the tables, in hours up to a day and in days beyond."""
    return f"{hours}\\,h" if hours <= 24 else f"{hours // 24}\\,d"


def format_p_value(p_value):
    """P value as a LaTeX cell, showing the resolution of the sign flip test rather than a zero."""
    return "$<0.001$" if p_value < 0.001 else f"${p_value:.3f}$"


def format_difference(difference, p_value):
    """Mean paired difference as a LaTeX cell, bold below BOLD_BELOW, with a fourth decimal when tiny."""
    text = f"{difference:+.4f}" if abs(difference) < 0.001 else f"{difference:+.3f}"

    return f"$\\mathbf{{{text}}}$" if p_value < BOLD_BELOW else f"${text}$"


def _padded_rows(rows):
    """Pad every column so the ampersands line up in the .tex source."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    return ["    " + " & ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() + " \\\\"
            for row in rows]


def generate_paired_table(results, horizons=TABLE_HORIZONS):
    """
    Print the reference error and the paired differences at the selected durations (tab:paired).

    Input
    -----
    results: DataFrame as returned by compare_outages
    horizons: Outage durations in hours to show (default is TABLE_HORIZONS)
    """
    tested = NUMBER_WORDS.get(results["Horizon"].nunique(), str(results["Horizon"].nunique()))
    windows = results["Windows"].max()

    reference = results[results["Reference"] == REFERENCE].groupby("Horizon")["Reference_RMSE"].first()
    rows = [["\\textbf{Contrast}"] + [f"\\textbf{{{duration_label(h)}}}" for h in horizons]]
    rows.append(["Neighbour regression, RMSE"] + [f"${reference[h]:.3f}$" for h in horizons])

    for model, other in CONTRASTS:
        pair = results[(results["Model"] == model) & (results["Reference"] == other)].set_index("Horizon")
        rows.append([f"{SHORT_NAMES[model]} $-$ {SHORT_NAMES[other]}"] + [
            format_difference(pair.loc[h, "Difference"], pair.loc[h, "P_holm"]) for h in horizons
        ])

    header, level, *contrasts = _padded_rows(rows)
    caption = (
        "Paired comparison of the three linear models: RMSE (\\textcelsius) of neighbour regression, and "
        f"mean paired RMSE difference of each pair of models over the same {windows} windows. A negative "
        "difference means that the first model of the pair is the more accurate one. Bold marks a "
        f"Holm-adjusted $p < {BOLD_BELOW}$ (paired permutation test, adjusted over the {tested} outage "
        f"durations). \\ref{{app:paired}} lists all {tested} durations with confidence intervals and $p$ values."
    )

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:paired}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * len(horizons)}@{{}}}}",
        "    \\toprule",
        f"    & \\multicolumn{{{len(horizons)}}}{{c}}{{\\textbf{{Outage duration}}}} \\\\",
        f"    \\cmidrule(l){{2-{len(horizons) + 1}}}",
        header,
        "    \\midrule",
        level,
        "    \\addlinespace",
        *contrasts,
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]

    print("\nLaTeX code for the paired comparison table:")
    print("\n".join(lines))


def generate_full_table(results):
    """
    Print every duration of every pair with its interval, p values and win rate (tab:paired_full).

    Input
    -----
    results: DataFrame as returned by compare_outages
    """
    windows = results["Windows"].max()
    rows = [["\\textbf{Duration}", "\\textbf{Difference}", "\\textbf{95\\,\\% CI}", "$\\boldsymbol{p}$",
             "$\\boldsymbol{p}_{\\mathrm{Holm}}$", "\\textbf{Wins}"]]
    groups = []

    for model, other in CONTRASTS:
        pair = results[(results["Model"] == model) & (results["Reference"] == other)].sort_values("Horizon")
        groups.append((len(rows), f"{MODEL_NAMES[model]} $-$ {MODEL_NAMES[other]}"))

        for _, row in pair.iterrows():
            rows.append([
                f"\\quad {duration_label(int(row['Horizon']))}",
                f"${row['Difference']:+.4f}$",
                f"$[{row['Lower']:+.4f}, {row['Upper']:+.4f}]$",
                format_p_value(row["P_value"]),
                format_p_value(row["P_holm"]),
                f"${row['Win_rate']:.2f}$",
            ])

    formatted = _padded_rows(rows)
    body = []
    for index, line in enumerate(formatted[1:], start=1):
        for start, heading in groups:
            if start == index:
                if body:
                    body.append("    \\addlinespace")
                body.append(f"    \\multicolumn{{6}}{{l}}{{\\emph{{{heading}}}}} \\\\")
        body.append(line)

    caption = (
        f"Paired RMSE differences (\\textcelsius) over {windows} windows: mean difference, 95\\,\\% percentile "
        "bootstrap confidence interval, raw and Holm-adjusted $p$ value of the paired permutation test, and "
        "the share of windows in which the first model of the pair is the more accurate."
    )

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:paired_full}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}",
        "    \\toprule",
        formatted[0],
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]

    print("\nLaTeX code for the full paired comparison table:")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    results = compare_outages(load_results(args.results_dir))

    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))

    generate_paired_table(results)
    generate_full_table(results)
