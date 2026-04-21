import numpy as np
import matplotlib.pyplot as plt

# Three evaluation scenarios
labels = ["Baseline\n(Regular Tor)", "Obfs4\n(Obfuscated)", "Zero-Shot\n(Unseen)"]

# Final model accuracies
rf_accuracies = [35.11, 17.14, 15.62]
triplet_accuracies = [36.44, 25.24, 15.14]

x = np.arange(len(labels))  # X positions for grouped bars
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars (blue: RF, green: Triplet)
rects1 = ax.bar(
    x - width / 2,
    rf_accuracies,
    width,
    label="Random Forest (Classic ML)",
    color="#1f77b4",
)
rects2 = ax.bar(
    x + width / 2,
    triplet_accuracies,
    width,
    label="Triplet Network (N-shot DL)",
    color="#2ca02c",
)

# Labels and styling
ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
ax.set_title("Website Fingerprinting Model Comparison", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 50)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.7)


# Add percentage labels above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("ml_eredmenyek_poc.png", dpi=300, bbox_inches="tight")
print("Chart saved: ml_eredmenyek_poc.png")
