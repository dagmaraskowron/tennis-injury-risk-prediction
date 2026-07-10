import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

TENNIS_GREEN = "#B7E4C7"
DARK_GREEN = "#40916C"


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
    ax = sns.barplot(
        data=results,
        x="Model",
        y="Dokładność",
        color=TENNIS_GREEN
    )

    plt.title("Porównanie dokładności modeli")
    plt.xlabel("Model")
    plt.ylabel("Dokładność")
    plt.ylim(0, 1.05)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3)

    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "10_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
