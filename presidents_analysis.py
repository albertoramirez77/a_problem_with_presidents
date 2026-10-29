"""
U.S. Presidents Lifespan Analysis

Actions
  1. Loads and cleans presidents CSV.
  2. Creates four new variables: year_of_birth, lived_years, lived_months, lived_days.
  3. Builds the two top-10 tables of longest-lived and shortest-lived presidents.
  4. Calculates summary statistics for lived_days.
  5. Plots the distribution of lifespans.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILE_PATH = "U.S. Presidents Birth and Death Information - Sheet1.csv"
OUTPUT_FOLDER = "output"

# Assume a fixed analysis date for simplicity in this case, +/- a few days does not GREATLY affect results
ANALYSIS_DATE = pd.Timestamp("2026-10-29")

DAYS_PER_YEAR = 365.25  # including leap-year
DAYS_PER_MONTH = DAYS_PER_YEAR / 12
TOP_N = 10

# I noticed that the CSV was retrieved on Sept 28, 2021, so it is missing one recent death.
DEATH_DATE_CORRECTIONS = {
    "Jimmy Carter": "Dec 29, 2024",
}

# Weights for the weighted average as the number of presidencies each person held.
PRESIDENCIES_SERVED_EXCEPTIONS = {
    "Grover Cleveland": 2,
    "Donald Trump": 2,
}

# Plot colors
COLOR_DECEASED = "#2a78d6"   # searched up these color values - blue
COLOR_LIVING = "#eb6834"     # orange
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"
COLOR_BACKGROUND = "#fcfcfb"


# Load data and standardize
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)   # create output/ if it doesn't exist
presidents = pd.read_csv(CSV_FILE_PATH)

# "BIRTH DATE" -> "birth_date", so every column follows the same style
presidents.columns = (
    presidents.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Remove rows with no birth date. This drops the "Reference:" citation
# row at the bottom of the file, which is not a president.
presidents = presidents.dropna(subset=["president", "birth_date"])

# Apply our given data correction of Jimmy Carter
for president_name, corrected_death_date in DEATH_DATE_CORRECTIONS.items():
    is_this_president = presidents["president"] == president_name
    presidents.loc[is_this_president, "death_date"] = corrected_death_date

# Data uses two types of month standardizations of July spelled
# out fully and Sep shorthanded, account for that
presidents["birth_date"] = pd.to_datetime(presidents["birth_date"], format="mixed")
presidents["death_date"] = pd.to_datetime(
    presidents["death_date"], format="mixed", errors="coerce"
)

# Safety check: every president must have a birth date, and the
# only missing death dates should belong to living presidents.
assert presidents["birth_date"].notna().all(), "Some birth dates failed to parse"
assert len(presidents) == 45, f"Expected 45 presidents, found {len(presidents)}"
assert presidents["death_date"].isna().sum() == 5, "Expected exactly 5 living presidents"

# Decide end data for each president
presidents["is_alive"] = presidents["death_date"].isna()
presidents["end_date"] = presidents["death_date"].fillna(ANALYSIS_DATE)

# Create the new variables from the ReadMe
presidents["year_of_birth"] = presidents["birth_date"].dt.year
presidents["lived_days"] = (presidents["end_date"] - presidents["birth_date"]).dt.days
presidents["lived_years"] = (presidents["lived_days"] / DAYS_PER_YEAR).round(2)
presidents["lived_months"] = (presidents["lived_days"] / DAYS_PER_MONTH).round(1)

presidents["presidencies_served"] = (
    presidents["president"].map(PRESIDENCIES_SERVED_EXCEPTIONS).fillna(1).astype(int)
)

# Construct the ranking tables
ranking_columns = [
    "president",
    "year_of_birth",
    "lived_years",
    "lived_months",
    "lived_days",
    "is_alive",
]


def rank_presidents_by_lifespan(data, longest_first):
    """Return the top N presidents sorted by lifespan, ranked starting at 1."""
    sorted_presidents = data.sort_values("lived_days", ascending=not longest_first)
    top_presidents = sorted_presidents.head(TOP_N)[ranking_columns].reset_index(drop=True)
    top_presidents.index = top_presidents.index + 1
    top_presidents.index.name = "rank"
    return top_presidents


longest_lived_presidents = rank_presidents_by_lifespan(presidents, longest_first=True)
shortest_lived_presidents = rank_presidents_by_lifespan(presidents, longest_first=False)

# Summary statistics
lived_days = presidents["lived_days"]

# Mode with edge case handling
lived_days_counts = lived_days.value_counts()
if lived_days_counts.max() == 1:
    lived_days_mode = "None (all values unique)"
else:
    lived_days_mode = ", ".join(str(value) for value in lived_days.mode())

summary_statistics = pd.DataFrame(
    {
        "statistic": [
            "Mean",
            "Weighted average",
            "Median",
            "Mode",
            "Maximum",
            "Minimum",
            "Standard deviation",
        ],
        "lived_days": [
            round(lived_days.mean(), 1),
            round(np.average(lived_days, weights=presidents["presidencies_served"]), 1),
            lived_days.median(),
            lived_days_mode,
            lived_days.max(),
            lived_days.min(),
            round(lived_days.std(ddof=0), 1),
        ],
    }
).set_index("statistic")

# Add a years column so the numbers are easier to read
summary_statistics["lived_years"] = [
    round(value / DAYS_PER_YEAR, 2) if not isinstance(value, str) else "—"
    for value in summary_statistics["lived_days"]
]

# Step 8: Plot the distribution (histogram)
#   A histogram shows how many presidents fall into each lifespan range.
#   The x-axis shows lived_days converted to years so it is readable.

five_year_bins = np.arange(45, 110, 5)
deceased_lived_years = presidents.loc[~presidents["is_alive"], "lived_days"] / DAYS_PER_YEAR
living_lived_years = presidents.loc[presidents["is_alive"], "lived_days"] / DAYS_PER_YEAR
mean_years = lived_days.mean() / DAYS_PER_YEAR
median_years = lived_days.median() / DAYS_PER_YEAR

figure, axis = plt.subplots(figsize=(9, 5.2), facecolor=COLOR_BACKGROUND)
axis.set_facecolor(COLOR_BACKGROUND)

axis.hist(
    [deceased_lived_years, living_lived_years],
    bins=five_year_bins,
    stacked=True,
    color=[COLOR_DECEASED, COLOR_LIVING],
    label=["Deceased", "Living (age as of Oct 29, 2026)"],
    edgecolor=COLOR_BACKGROUND,
    linewidth=2,
)

axis.axvline(mean_years, color=COLOR_TEXT, linewidth=1.5, linestyle="--")
axis.axvline(median_years, color=COLOR_TEXT, linewidth=1.5, linestyle=":")
# Mean should be slightly larger than median, so its label goes on the right
axis.text(mean_years + 0.5, 9.6, f"Mean {mean_years:.1f}", ha="left",
          color=COLOR_TEXT, fontsize=9)
axis.text(median_years - 0.5, 9.6, f"Median {median_years:.1f}", ha="right",
          color=COLOR_TEXT, fontsize=9)

axis.set_title("Distribution of U.S. Presidents' Lifespans (n = 45)",
               loc="left", fontsize=13, color=COLOR_TEXT, pad=12)
axis.set_xlabel("Years lived (lived_days ÷ 365.25), 5-year bins",
                color=COLOR_TEXT_MUTED)
axis.set_ylabel("Number of presidents", color=COLOR_TEXT_MUTED)
axis.set_xticks(five_year_bins)
axis.set_ylim(0, 10.5)
axis.yaxis.grid(True, color=COLOR_GRID, linewidth=1)
axis.set_axisbelow(True)
axis.tick_params(colors=COLOR_TEXT_MUTED)
for side in ["top", "right", "left"]:
    axis.spines[side].set_visible(False)
axis.spines["bottom"].set_color(COLOR_TEXT_MUTED)
axis.legend(frameon=False, loc="upper left", labelcolor=COLOR_TEXT)

figure.tight_layout()
figure.savefig(f"{OUTPUT_FOLDER}/lifespan_distribution.png", dpi=200)

# Print and save each table
longest_lived_presidents.to_csv(f"{OUTPUT_FOLDER}/longest_lived_presidents.csv")
shortest_lived_presidents.to_csv(f"{OUTPUT_FOLDER}/shortest_lived_presidents.csv")
summary_statistics.to_csv(f"{OUTPUT_FOLDER}/lived_days_summary_statistics.csv")
presidents.to_csv(f"{OUTPUT_FOLDER}/presidents_with_lifespans.csv", index=False)

print("Top 10 Longest-Lived Presidents")
print(longest_lived_presidents.to_markdown())
print()
print("Top 10 Shortest-Lived Presidents")
print(shortest_lived_presidents.to_markdown())
print()
print("Summary Statistics for lived_days")
print(summary_statistics.to_markdown())