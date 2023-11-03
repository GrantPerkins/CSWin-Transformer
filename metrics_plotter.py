from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


base = Path(r"output/finetune/20231031-152142-CSWin_64_12211_tiny_224-224")
out=Path("model_metrics/CSWin_64_12211_tiny_224")
df = pd.read_csv(str(base/"summary.csv"))
print(df)

# plot loss
df[["train_loss", "eval_loss"]].plot()
plt.title("Train vs. Evaluation Loss")
plt.grid()
plt.ylabel("Cross-Entropy Loss")
plt.xlabel("Epochs")
# plt.show()
plt.savefig(str(out/"losses.png"))
plt.close()
# plot accuracy
df[["train_top1", "eval_top1"]].plot()
plt.title("Train vs. Evaluation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.savefig(str(out/"accuracies.png"))
# plt.show()
plt.close()