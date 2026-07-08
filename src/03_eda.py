import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_tennis_biomechanics.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

RISK_ORDER = ["Low", "Medium", "High"]

TENNIS_GREEN = "#B7E4C7"
DARK_GREEN = "#40916C"
LIGHT_GREEN = "#D8F3DC"
YELLOW_GREEN = "#95D5B2"

RISK_PALETTE = {
    "Low": LIGHT_GREEN,
    "Medium": YELLOW_GREEN,
    "High": DARK_GREEN
}

SLIP_PALETTE = {
    0: LIGHT_GREEN,
    1: DARK_GREEN
}


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:")
    print(df.shape)

    print("\nInjury risk distribution:")
    print(df["Injury_Risk_Level"].value_counts())

    print("\nCategorical columns:")
    print(df.select_dtypes(include=["object", "string"]).columns.tolist())

    print("\nNumeric columns:")
    print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df,
        x="Injury_Risk_Level",
        order=RISK_ORDER,
        hue="Injury_Risk_Level",
        palette=RISK_PALETTE,
        legend=False
    )
    plt.title("Distribution of Injury Risk Level")
    plt.xlabel("Injury Risk Level")
    plt.ylabel("Number of Observations")
    save_plot("01_injury_risk_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Injury_Risk_Level",
        y="Grip_Stability_Score",
        order=RISK_ORDER,
        hue="Injury_Risk_Level",
        palette=RISK_PALETTE,
        legend=False
    )
    plt.title("Grip Stability Score by Injury Risk Level")
    plt.xlabel("Injury Risk Level")
    plt.ylabel("Grip Stability Score")
    save_plot("02_grip_stability_by_risk.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Injury_Risk_Level",
        y="Muscle_Fatigue_Index",
        order=RISK_ORDER,
        hue="Injury_Risk_Level",
        palette=RISK_PALETTE,
        legend=False
    )
    plt.title("Muscle Fatigue Index by Injury Risk Level")
    plt.xlabel("Injury Risk Level")
    plt.ylabel("Muscle Fatigue Index")
    save_plot("03_muscle_fatigue_by_risk.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Injury_Risk_Level",
        y="Impact_Force_N",
        order=RISK_ORDER,
        hue="Injury_Risk_Level",
        palette=RISK_PALETTE,
        legend=False
    )
    plt.title("Impact Force by Injury Risk Level")
    plt.xlabel("Injury Risk Level")
    plt.ylabel("Impact Force (N)")
    save_plot("04_impact_force_by_risk.png")

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df,
        x="Injury_Risk_Level",
        hue="Slip_Event",
        order=RISK_ORDER,
        palette=SLIP_PALETTE
    )
    plt.title("Slip Events by Injury Risk Level")
    plt.xlabel("Injury Risk Level")
    plt.ylabel("Number of Observations")
    plt.legend(title="Slip Event", labels=["No", "Yes"])
    save_plot("05_slip_events_by_risk.png")

    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=df,
        x="Stroke_Type",
        hue="Injury_Risk_Level",
        hue_order=RISK_ORDER,
        palette=RISK_PALETTE
    )
    plt.title("Injury Risk Level by Stroke Type")
    plt.xlabel("Stroke Type")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=30, ha="right")
    save_plot("06_injury_risk_by_stroke_type.png")

    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=df,
        x="Surface_Type",
        hue="Injury_Risk_Level",
        hue_order=RISK_ORDER,
        palette=RISK_PALETTE
    )
    plt.title("Injury Risk Level by Surface Type")
    plt.xlabel("Surface Type")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=30, ha="right")
    save_plot("07_injury_risk_by_surface_type.png")

    selected_features = [
        "Grip_Force_Mean",
        "Grip_Force_SD",
        "Peak_Grip_Force",
        "Pressure_Distribution_Index",
        "Contact_Area_cm2",
        "EMG_Flexor_RMS",
        "EMG_Extensor_RMS",
        "Muscle_Fatigue_Index",
        "Impact_Force_N",
        "Vibration_Peak",
        "Vibration_Frequency_Hz",
        "Shock_Transmission_Index",
        "Grip_Stability_Score",
        "Slip_Event"
    ]

    plt.figure(figsize=(13, 10))
    sns.heatmap(
        df[selected_features].corr(),
        cmap="Greens",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap of Biomechanical Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_plot("08_correlation_heatmap.png")

    print("\nEDA completed. Plots saved in images folder.")