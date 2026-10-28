import glob
import pandas as pd

RESULTS_GLOB = "output/interval_coverage/interval_coverage_*.csv"

# Hours into the outage at which the table splits its coverage, so a shortfall late in long outages shows up
# separately from the first two weeks
SPLIT_HOURS = 336


def pooled_coverage(df, group_cols):
    """
    Coverage as the ratio of the pooled hour counts, not the mean of the per-window ratios, so short
    and long windows are weighted correctly.

    Input
    -----
    df: DataFrame with N_inside and N_total
    group_cols: Columns to group by

    Output
    ------
    summary: Group columns plus a Coverage column
    """
    grouped = df.groupby(group_cols, as_index=False)[["N_inside", "N_total"]].sum()
    grouped["Coverage"] = grouped["N_inside"] / grouped["N_total"]

    return grouped


def print_coverage_table(df):
    """
    Pooled coverage of the 95 % interval per band method and validation episode, over the whole outage, its first
    SPLIT_HOURS and the hours after, with the mean interval width, as in the paper's coverage table.

    Input
    -----
    df: Coverage rows of every dataset, station, onset, method, episode and horizon
    """
    longest = df["Horizon"].max()
    summary = pooled_coverage(df, ["Method", "Calibration_days", "Horizon"])
    widths = df.groupby(["Method", "Calibration_days"])["Mean_width"].mean()

    print(f"{'method':<11}{'episode':>9}{'1-' + str(longest // 24) + 'd':>9}{'1-' + str(SPLIT_HOURS // 24) + 'd':>9}"
          f"{str(SPLIT_HOURS // 24 + 1) + '-' + str(longest // 24) + 'd':>9}{'width':>9}")

    for (method, days), rows in summary.groupby(["Method", "Calibration_days"]):
        end = rows[rows["Horizon"] == longest].iloc[0]
        split = rows[rows["Horizon"] == SPLIT_HOURS].iloc[0]

        # The counts of a horizon cover every hour of the outage up to it, so the later hours are the difference
        # between the longest horizon and the split
        late = (end["N_inside"] - split["N_inside"]) / (end["N_total"] - split["N_total"])

        print(f"{method:<11}{str(days) + 'd':>9}{end['Coverage']:>9.3f}{split['Coverage']:>9.3f}{late:>9.3f}"
              f"{widths.loc[(method, days)]:>9.2f}")


if __name__ == "__main__":
    print_coverage_table(pd.concat([pd.read_csv(f) for f in glob.glob(RESULTS_GLOB)], ignore_index=True))
