import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "clean_tennis_biomechanics.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

TARGET_COLUMN = "Injury_Risk_Level"


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)

    return X, y


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X, y = prepare_data(df)

    print("Features shape:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_model_name = None
    best_accuracy = 0

    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 50)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:")
        print(round(accuracy, 4))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]))

        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nBest model:")
    print(best_model_name)

    print("\nBest accuracy:")
    print(round(best_accuracy, 4))

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")