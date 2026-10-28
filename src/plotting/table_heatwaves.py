import numpy as np
import pandas as pd
from src.scripts.heatwave_experiment import OUTPUT_DIR, TRAINING_WEEKS

# Errors of the table, and the outage start of the table in days before the first heat wave day
METRICS = ["RMSE", "Bias", "Tmax_error", "Tmin_error"]
PAPER_LEAD = 3
PAPER_LEAD_TEXT = "three days"

KEYS = ["Dataset", "Station", "Event_id", "Control", "Lead_days"]

N_BOOTSTRAP = 10000
SEED = 42


def window_metrics(scored, window):
    """
    Errors of one window over its heat wave days, the reconstruction minus the observation.

    Input
    -----
    scored: Hourly rows of the window, with Datetime, Observed and Predicted
    window: Row of the window table

    Output
    ------
    metrics: Dict with one value per name in METRICS
    """
    scored = scored.set_index("Datetime")

    # The experiment also scores the days after the heat wave, which the table leaves out
    heat = scored[:window.Window_end + pd.Timedelta(hours=23)]
    error = heat.Predicted - heat.Observed
    heat_days = heat.groupby(heat.index.floor("D"))

    return {
        "RMSE": np.sqrt((error ** 2).mean()),
        "Bias": error.mean(),
        "Tmax_error": (heat_days.Predicted.max() - heat_days.Observed.max()).mean(),
        "Tmin_error": (heat_days.Predicted.min() - heat_days.Observed.min()).mean()
    }


def bootstrap_ci(values, rng):
    """
    Percentile bootstrap 95 % interval of the mean.

    Input
    -----
    values: Array with one value per event
    rng: Random generator

    Output
    ------
    low, high: Interval bounds
    """
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values))).mean(axis=1)

    return np.percentile(means, [2.5, 97.5])


def summarise(metrics):
    """
    Heat wave mean, control mean and their difference per dataset, lead and metric, with intervals over events.
    Each heat wave window is paired with the mean of its controls (same station, same lead).

    Input
    -----
    metrics: DataFrame with KEYS and METRICS, one row per window

    Output
    ------
    summary: DataFrame with one row per (dataset, lead, metric)
    """
    rng = np.random.default_rng(SEED)

    pair_keys = ["Dataset", "Station", "Event_id", "Lead_days"]
    heatwave = metrics[metrics.Control == 0].set_index(pair_keys)[METRICS]
    controls = metrics[metrics.Control > 0].groupby(pair_keys)[METRICS].mean()

    # Every event has the same stations, so the mean over events equals the mean over windows
    per_event = {
        name: frame.groupby(["Dataset", "Lead_days", "Event_id"]).mean()
        for name, frame in [("Heatwave", heatwave), ("Control", controls), ("Difference", heatwave - controls)]
    }

    rows = []

    for (dataset, lead), events in per_event["Heatwave"].groupby(level=["Dataset", "Lead_days"]):
        for metric in METRICS:
            row = {"Dataset": dataset, "Lead_days": lead, "Metric": metric, "N_events": len(events)}

            for name, frame in per_event.items():
                values = frame.loc[(dataset, lead), metric].values
                low, high = bootstrap_ci(values, rng)
                row.update({name: values.mean(), f"{name}_low": low, f"{name}_high": high})

            rows.append(row)

    return pd.DataFrame(rows)


def print_latex_table(summary):
    """
    Print the table of the paper as LaTeX: heat waves and ordinary weather as rows, the errors of METRICS as
    columns, at PAPER_LEAD, with the RMSE difference and its interval in the caption.

    Input
    -----
    summary: Output of summarise
    """
    at_lead = summary[summary.Lead_days == PAPER_LEAD].set_index(["Dataset", "Metric"])
    dataset_labels = {"TURKU": "TURCLIM", "SYNTHETIC": "Synthetic"}
    caption_labels = {"TURKU": "TURCLIM", "SYNTHETIC": "the synthetic network"}
    column_labels = {"RMSE": "RMSE", "Bias": "Mean bias", "Tmax_error": "Daily max.", "Tmin_error": "Daily min."}

    def format_value(metric, value):
        # Rounded first, so a bias of -0.003 prints as 0.00 rather than -0.00
        value = round(value, 2) + 0.0
        return f"${value:.2f}$" if metric == "RMSE" or value == 0 else f"${value:+.2f}$"

    header = ["\\textbf{Weather}"] + [f"\\textbf{{{column_labels[metric]}}}" for metric in METRICS]
    columns = len(header)

    body = []
    for dataset in dataset_labels:
        body.append(("group", f"{dataset_labels[dataset]} ({at_lead.loc[(dataset, 'RMSE'), 'N_events']} heat waves)"))
        for weather, column in [("Heat waves", "Heatwave"), ("Ordinary weather", "Control")]:
            body.append(("row", [f"\\quad {weather}"] + [
                format_value(metric, at_lead.loc[(dataset, metric), column]) for metric in METRICS
            ]))

    # Pad every column so the ampersands line up in the .tex source
    rows = [cells for kind, cells in body if kind == "row"]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(columns)]

    def format_row(cells):
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
        return "    " + " & ".join(padded).rstrip() + " \\\\"

    # Only the first interval names what it is
    differences = []
    for index, dataset in enumerate(dataset_labels):
        rmse = at_lead.loc[(dataset, "RMSE")]
        differences.append(
            f"by {rmse.Difference:.2f}\\,\\textcelsius{{}} in {caption_labels[dataset]} "
            f"({'95\\,\\% confidence interval ' if index == 0 else ''}"
            f"{rmse.Difference_low:.2f}--{rmse.Difference_high:.2f}\\,\\textcelsius{{}})"
        )

    caption = (
        "Accuracy of RegSARIMA during heat waves and in windows of ordinary weather around the same events, for an "
        f"outage that starts {PAPER_LEAD_TEXT} before the heat wave. Errors are the reconstruction minus the "
        "observation, in \\textcelsius{}, so a negative value means an underestimation. The errors in the daily "
        "maximum and minimum are averaged over the days of each window. During heat waves, the RMSE is higher "
        f"{differences[0]} and {differences[1]}."
    )

    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        "  \\label{tab:heatwaves}",
        "  \\small",
        f"  \\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}l{'r' * (columns - 1)}@{{}}}}",
        "    \\toprule",
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

    print("\nLaTeX code for the heat wave table:")
    print("\n".join(lines))


if __name__ == "__main__":
    hourly = pd.read_csv(
        f"{OUTPUT_DIR}/RegressionSARIMAErrors_{TRAINING_WEEKS}_weeks.csv.gz", parse_dates=["Datetime"]
    )
    windows = pd.read_csv(
        f"{OUTPUT_DIR}/windows_{TRAINING_WEEKS}_weeks.csv", parse_dates=["Window_start", "Window_end"]
    ).set_index(KEYS)

    metrics = pd.DataFrame([
        {**dict(zip(KEYS, keys)), **window_metrics(scored, windows.loc[keys])}
        for keys, scored in hourly.groupby(KEYS)
    ])
    summary = summarise(metrics)

    # Every outage start, for the statement that the result holds regardless of when the outage starts
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(summary[["Dataset", "Lead_days", "Metric", "N_events", "Heatwave", "Control", "Difference",
                       "Difference_low", "Difference_high"]].to_string(index=False))

    print_latex_table(summary)
