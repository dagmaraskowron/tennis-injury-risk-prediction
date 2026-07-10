import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_tennis_biomechanics.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Injury_Risk_Level"

TENNIS_GREEN = "#B7E4C7"


def prepare_data(df: pd.DataFrame):
    """Prepare features in the same way as during modeling."""
    X = df.drop(columns=[TARGET_COLUMN])
    X = pd.get_dummies(X, drop_first=True)

    return X


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = prepare_data(df)

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("The selected model does not provide feature_importances_.")

    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Top 10 most important features:")
    print(importances.head(10))

    importances.to_csv(
        PROJECT_ROOT / "models" / "feature_importance.csv",
        index=False
    )

    feature_labels = {
        "Vibration_Peak": "Szczyt wibracji",
        "Grip_Force_SD": "Zmienność siły chwytu",
        "Muscle_Fatigue_Index": "Indeks zmęczenia mięśni",
        "Contact_Area_cm2": "Pole kontaktu",
        "Temperature_C": "Temperatura",
        "Grip_Stability_Score": "Stabilność chwytu",
        "Coating_Roughness_nm": "Chropowatość powłoki",
        "Friction_Coefficient": "Współczynnik tarcia",
        "Peak_Grip_Force": "Maksymalna siła chwytu",
        "Age": "Wiek",
        "Grip_Force_Mean": "Średnia siła chwytu",
        "Humidity_%": "Wilgotność",
        "EMG_Extensor_RMS": "Aktywność prostowników",
        "EMG_Flexor_RMS": "Aktywność zginaczy",
        "Pressure_Distribution_Index": "Indeks rozkładu nacisku",
        "Impact_Force_N": "Siła uderzenia",
        "Shock_Transmission_Index": "Indeks przenoszenia wstrząsu",
        "Vibration_Frequency_Hz": "Częstotliwość wibracji",
        "Slip_Event": "Poślizg chwytu"
    }

    top_features = importances.head(5).copy()
    top_features["Feature_Label"] = top_features["Feature"].map(feature_labels)
    top_features["Feature_Label"] = top_features["Feature_Label"].fillna(top_features["Feature"])

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature_Label",
        color=TENNIS_GREEN
    )

    plt.title("Najważniejsze cechy modelu")
    plt.xlabel("Ważność cechy")
    plt.ylabel("")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "09_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

