import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("checkpoints/medium/training_log.csv")

plt.figure(figsize=(8, 4))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("outputs/training_progress.png")
