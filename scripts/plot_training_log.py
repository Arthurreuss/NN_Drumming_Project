import os

import matplotlib.pyplot as plt
import pandas as pd

# Load training log
df = pd.read_csv("checkpoints/seg_64/medium/training_log.csv")

# Create output directory if missing
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
plt.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("outputs/training_progress.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
