"""
Group of functions to print the paired comparison of ARIMAX, the neighbour regression and the
two-stage regression with SARIMA errors (src/evaluation/paired_comparison.py).

python -m src.plotting.plot_paired_comparison
"""
from src.evaluation.paired_comparison import compare
from src.plotting.plot_short_horizon_comparison import load_results, MODEL_NAMES, OUTPUT_DIR

# Pairs of models to compare, the neighbour regression as the reference for the two models that
# extend it, plus the two error models against each other
CONTRASTS = [
    ("RegressionSARIMAErrors", "LinearRegression"),
    ("ARIMAX", "LinearRegression"),
    ("RegressionSARIMAErrors", "ARIMAX"),
]

# Forecast hours the comparison is run at
LEADS = [1, 2, 4, 8, 12, 24, 48, 96]

# Forecast hours the table shows, a subset because a column per hour does not fit the width of a
# page. The comparison itself still runs over every hour of LEADS, so the adjusted p values of the
# table are the ones of the full family and the appendix can report the hours left out
TABLE_LEADS = [1, 2, 4, 8, 24, 96]

# Model the differences are taken against, chosen as the simplest of the three rather than the most
# accurate, so no contrast is biased by the reference having been picked on its score
REFERENCE = "LinearRegression"

# Row label of the reference level and of every contrast, with the models named in full in the
# caption so the labels stay short enough to leave a column per forecast hour
REFERENCE_LABEL = "Neighbour reg., MAE"
CONTRAST_LABELS = {
    ("RegressionSARIMAErrors", "LinearRegression"): "Two-stage $-$ Neighb.",
    ("ARIMAX", "LinearRegression"): "ARIMAX $-$ Neighb.",
    ("RegressionSARIMAErrors", "ARIMAX"): "Two-stage $-$ ARIMAX",
}

# Adjusted p values below which a difference is set in bold and below which it also gets a star
BOLD_BELOW = 0.05
STAR_BELOW = 0.001


def contrast_label(model, reference):
    """
    Name of a contrast as used in the plots and the printed tables.

    Input
    -----
    model: Key of the model being compared
    reference: Key of the model it is compared against

    Output
    ------
    Returns the two model names separated by a minus sign.
    """
    return f"{MODEL_NAMES[model]} - {MODEL_NAMES[reference]}"


def format_p_value(p_value):
    """
    Format a p value, showing the resolution of the sign flip test rather than a misleading zero.

    Input
    -----
    p_value: The raw or the adjusted p value

    Output
    ------
    Returns the p value as a string.
    """
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def print_comparison(results, title):
    """
    Print one comparison as a table, one line per contrast and forecast hour.

    Input
    -----
    results: DataFrame as returned by compare
    title: Heading printed above the table
    """
    print(f"\n=== {title} ===")
    print("The interval covers one contrast at one forecast hour, the adjusted p value covers every "
          "forecast hour of that contrast, so the two can disagree")
    print(f"{'Leads':>8}{'Difference':>13}{'95% interval':>22}{'p':>9}{'p (Holm)':>11}"
          f"{'Wins':>7}   Contrast")

    for _, row in results.iterrows():
        interval = f"[{row['Lower']:+.4f}, {row['Upper']:+.4f}]"
        print(f"{row['Leads']:>8}{row['Difference']:>+13.4f}{interval:>22}"
              f"{format_p_value(row['P_value']):>9}{format_p_value(row['P_holm']):>11}"
              f"{row['Win_rate']:>7.0%}   "
              f"{contrast_label(row['Model'], row['Reference'])}")


def format_difference(difference, p_value):
    """
    Format a paired difference as a math mode LaTeX cell, marking how strong the evidence is.

    Input
    -----
    difference: The mean paired difference in degrees Celsius
    p_value: The adjusted p value of that difference

    Output
    ------
    Returns the difference in bold below BOLD_BELOW and with an added star below STAR_BELOW.
    """
    text = f"{difference:+.3f}"

    if p_value < STAR_BELOW:
        return f"$\\mathbf{{{text}}}^{{\\ast}}$"
    if p_value < BOLD_BELOW:
        return f"$\\mathbf{{{text}}}$"

    return f"${text}$"


def generate_paired_table(panels):
    """
    Print the paired differences as a booktabs LaTeX table, one panel per way of averaging the error.

    Input
    -----
    panels: List of (heading, results) tuples, with results as returned by compare
    """
    columns = len(TABLE_LEADS) + 1
    header = ["\\textbf{Contrast}"] + [f"\\textbf{{{lead}}}" for lead in TABLE_LEADS]

    # Either ("group", heading) or ("row", cells), in table order
    body = []
    for heading, results in panels:
        body.append(("group", heading))

        # The level of the reference, so the differences below it can be read against a scale
        levels = results[results["Reference"] == REFERENCE].groupby("Leads")["Reference_MAE"].first()
        body.append(("row", [f"\\quad {REFERENCE_LABEL}"]
                     + [f"${levels[lead]:.3f}$" for lead in TABLE_LEADS]))

        for model, reference in CONTRASTS:
            contrast = results[(results["Model"] == model) & (results["Reference"] == reference)]
            contrast = contrast.set_index("Leads")
            body.append(("row", [f"\\quad {CONTRAST_LABELS[(model, reference)]}"] + [
                format_difference(contrast.loc[lead, "Difference"], contrast.loc[lead, "P_holm"])
                for lead in TABLE_LEADS
            ]))

    # Pad every column so the ampersands line up in the .tex source
    rows = [cells for kind, cells in body if kind == "row"]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(columns)]

    def format_row(cells):
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
        return "    " + " & ".join(padded).rstrip() + " \\\\"

    caption = (
        "Mean absolute error (\\textcelsius) of the neighbour regression and the paired difference "
        "of the two models that extend it, over short outages, averaged over 16 stations and 160 "
        "failure onsets. Two-stage is the regression with SARIMA errors and Neighb. the neighbour "
        "regression; a negative difference means the first model of the contrast is the more "
        f"accurate one. Bold marks a difference with a Holm adjusted $p < {BOLD_BELOW}$, a star "
        f"one with $p < {STAR_BELOW}$. The comparison runs over {len(LEADS)} forecast hours and "
        "the adjustment covers all of them; the appendix reports the hours left out here, with the "
        "confidence intervals and the $p$ values behind the marks."
    )

    lines = [
        "\\begin{table}[htb]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:paired}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * (columns - 1)}@{{}}}}",
        "    \\toprule",
        f"    & \\multicolumn{{{len(TABLE_LEADS)}}}{{c}}{{\\textbf{{Forecast hour}}}} \\\\",
        f"    \\cmidrule(l){{2-{columns}}}",
        format_row(header),
        "    \\midrule",
    ]
    for index, (kind, value) in enumerate(body):
        if kind == "group":
            if index:
                lines.append("    \\addlinespace")
            lines.append(f"    \\multicolumn{{{columns}}}{{l}}{{\\emph{{{value}}}}} \\\\")
        else:
            lines.append(format_row(value))
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]

    print("\nLaTeX code for the paired comparison table:")
    print("\n".join(lines))


if __name__ == "__main__":
    # Load all the csv files generated by the short horizon comparison experiment
    df = load_results(OUTPUT_DIR)

    print(f"{len(df)} rows, {df['Station'].nunique()} stations, "
          f"{df['Train_start'].nunique()} forecast starts")

    # The error at one forecast hour, which is where the models actually differ
    per_lead = compare(df, CONTRASTS, [(lead, [lead]) for lead in LEADS])
    print_comparison(per_lead, "Paired difference at one forecast hour (°C)")

    # The error averaged over the whole forecast, matching the cumulative figure
    cumulative = compare(df, CONTRASTS, [(lead, range(1, lead + 1)) for lead in LEADS])
    print_comparison(cumulative, "Paired difference over the whole forecast window (°C)")

    generate_paired_table([
        ("Error at forecast hour $h$", per_lead),
        ("Error averaged over hours 1 to $h$", cumulative),
    ])
