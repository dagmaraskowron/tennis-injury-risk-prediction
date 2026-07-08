import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    results = pd.DataFrame({
        "Model": [
            "Regresja logistyczna",
            "Las losowy"
        ],
        "Dokładność": [
            0.7134,
            0.9969
        ]
    })

    print(results)

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=results,
        x="Model",
        y="Dokładność",
        color="#B7E4C7"
    )
    plt.title("Porównanie dokładności modeli")
    plt.xlabel("Model")
    plt.ylabel("Dokładność")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "10_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nModel comparison plot saved to:")
    print(IMAGES_DIR / "10_model_comparison.png")