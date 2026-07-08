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


def prepare_data(df: pd.DataFrame):
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

    print("Top 15 most important features:")
    print(importances.head(15))

    importances.to_csv(
        PROJECT_ROOT / "models" / "feature_importance.csv",
        index=False
    )

    top_features = importances.head(3).copy()

    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        color = "#B7E4C7"
    )
    plt.title("Top 3 Most Important Features")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "09_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()