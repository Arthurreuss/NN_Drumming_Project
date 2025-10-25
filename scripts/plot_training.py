from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Lade alle Logs

csv_files = (
    list(Path("checkpoints/seg_512/l16/medium/0.001").rglob("training_log*.csv"))
    + list(Path("checkpoints/seg_512/l16/small/0.001").rglob("training_log*.csv"))
    + list(Path("checkpoints/seg_512/l16/large/0.001").rglob("training_log*.csv"))
)
logs = {}

for path in csv_files:
    try:
        df = pd.read_csv(path)
        if "epoch" in df.columns:
            last_two = path.parent.parts[-2:]
            logs[f"model: {last_two[0]}"] = df
        else:
            print(f"Skipping {path.name}, no 'epoch' column.")
    except Exception as e:
        print(f"Failed to read {path}: {e}")

# Step 2: Plot nur val/train loss pro LR
output_dir = Path("outputs/model_comparisons")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 7))

# Farb-Cycler vorbereiten
color_cycler = plt.cm.get_cmap("tab10")  # oder z.B. "Set1", "Dark2", etc.
colors = {}

for idx, (name, df) in enumerate(logs.items()):
    color = color_cycler(idx % 10)  # modulo für Wiederholung bei mehr als 10 LRs
    colors[name] = color

    if "train_loss" in df.columns:
        plt.plot(
            df["epoch"],
            df["train_loss"],
            label=f"{name} (train)",
            linestyle="-",
            color=color,
        )
    if "val_loss" in df.columns:
        plt.plot(
            df["epoch"],
            df["val_loss"],
            label=f"{name} (val)",
            linestyle="--",
            color=color,
        )

plt.title("Train & Validation Loss across Models")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output_dir / "model_comparison_all_models.png")
plt.close()

print(f"Plot saved to: {output_dir / 'loss_comparison_all_models.png'}")
