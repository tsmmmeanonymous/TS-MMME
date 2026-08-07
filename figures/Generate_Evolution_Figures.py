import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.dates as mdates

# ======================================================
# Publication Color Palette
# ======================================================

COLOR_EVOLUTION = "#1f77b4"   # Blue
COLOR_TRUTH     = "#2ca02c"   # Green
COLOR_RUMOR     = "#d62728"   # Red
COLOR_RETRIEVAL = "#ff7f0e"   # Orange


# -------------------------------------------------------
# Paths
# -------------------------------------------------------


# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Search recursively for every evolution timeline
timeline_files = list(PROJECT_ROOT.rglob("evolution_timeline.csv"))

if len(timeline_files) == 0:
    raise FileNotFoundError("No evolution_timeline.csv files found.")

print(f"Found {len(timeline_files)} timeline files.")

OUTPUT_DIR = PROJECT_ROOT / "Plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

all_df = []

for file in timeline_files:
    temp = pd.read_csv(file)
    temp["Source"] = file.parent.name
    all_df.append(temp)

df = pd.concat(all_df, ignore_index=True)

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")

plt.style.use("ggplot")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# =======================================================
# Figure 3 : Temporal Evolution of Veracity
# =======================================================

# =======================================================
# Aggregate Monthly Evolution
# =======================================================

monthly = (
    df.groupby("Month")["Evolution_Score"]
      .mean()
      .reset_index()
)

monthly["Month"] = pd.to_datetime(monthly["Month"])
monthly = monthly.sort_values("Month")

plt.figure(figsize=(10,5))

plt.plot(
    monthly["Month"],
    monthly["Evolution_Score"],
    color=COLOR_EVOLUTION,
    linewidth=2.2
)

plt.fill_between(
    monthly["Month"],
    monthly["Evolution_Score"],
    alpha=0.20,
    color=COLOR_EVOLUTION
)

plt.scatter(
    monthly["Month"],
    monthly["Evolution_Score"],
    color=COLOR_EVOLUTION,
    s=25,
    zorder=3
)

plt.title("Temporal Evolution of Veracity")
plt.xlabel("Month")
plt.ylabel("Evolution Score")
plt.grid(True, alpha=0.3)

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(
    OUTPUT_DIR/"Figure3_Temporal_Evolution.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close()

# =======================================================
# Figure 4 : Truth vs Rumor Evolution
# =======================================================

monthly = (
    df.groupby("Month")[["Truth_Ratio", "Rumor_Ratio"]]
      .mean()
      .reset_index()
)

monthly["Month"] = pd.to_datetime(monthly["Month"])
monthly = monthly.sort_values("Month")

plt.figure(figsize=(10,5))

plt.plot(
    monthly["Month"],
    monthly["Truth_Ratio"],
    color=COLOR_TRUTH,
    linewidth=2.2,
    marker="o",
    markevery=6,
    markersize=4,
    label="Truth Ratio"
)

plt.plot(
    monthly["Month"],
    monthly["Rumor_Ratio"],
    color=COLOR_RUMOR,
    linewidth=2.2,
    marker="s",
    markevery=6,
    markersize=4,
    label="Rumor Ratio"
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5,1.22),
    ncol=2,
    frameon=False
)

plt.xlabel("Month")
plt.ylabel("Ratio")
plt.title("Truth vs Rumor Evolution")

plt.grid(True, alpha=0.3)

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(
    OUTPUT_DIR/"Figure4_Truth_Rumor_Evolution.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close()

# =======================================================
# Figure 5 : Spatial Evolution Analysis
# =======================================================

counts = df["Location"].value_counts()

valid_locations = counts[counts >= 20].index

location_score = (
    df[df["Location"].isin(valid_locations)]
      .groupby("Location")["Evolution_Score"]
      .mean()
      .sort_values(key=lambda s: s.abs(), ascending=False)
      .head(20)
      .sort_values()
)

plt.figure(figsize=(10,9))

location_score.plot(
    kind="barh",
    color=plt.cm.viridis(
        np.linspace(0.2,0.9,len(location_score))
    )
)

plt.xlabel("Average Evolution Score")
plt.ylabel("Location")
plt.yticks(fontsize=9)
plt.title("Spatial Evolution Analysis")

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(
    OUTPUT_DIR/"Figure5_Spatial_Evolution.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close()

# =======================================================
# Figure 6 : Evolution Score Distribution
# =======================================================

plt.figure(figsize=(7,4.5))

plt.hist(
    df["Evolution_Score"],
    bins=40,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85
)

plt.xlabel("Evolution Score")
plt.ylabel("Frequency")
plt.title("Distribution of Evolution Scores")

plt.grid(
    linestyle="--",
    alpha=0.35
)

plt.tight_layout()

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(
    OUTPUT_DIR/"Figure6_Evolution_Distribution.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.15
)

plt.close()

print("="*60)
print("Evolution Figures Generated Successfully")
print("="*60)
print("Saved to :", OUTPUT_DIR.resolve())