import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

fold_df = {"fold": [], "Evaluation Accuracy": [], "Training Accuracy": [], "Stage 1": [], "Stage 2": [], "Stage 3": [], "Stage 4": []}

base_path = Path(r"C:\Users\gcper\Code\Thesis\CSWin-Transformer\model_metrics")
for path in Path(r"C:\Users\gcper\Code\Thesis\CSWin-Transformer\output\finetune").iterdir():
    if (path / "args.yaml").exists():
        with open(str(path / "args.yaml"), 'r') as f:
            args = yaml.safe_load(f)
        if args["epochs"] == 250:
            fold = args["val_fold"]
            summary_path = path / "summary.csv"
            df = pd.read_csv(str(summary_path))
            fold_df["fold"].append(fold)
            fold_df["Evaluation Accuracy"].append(max(df["eval_top1"]))
            fold_df["Training Accuracy"].append(100*max(df["train_top1"]))
            df = pd.read_csv(base_path/f"CSWin_64_12211_tiny_224_fold_{fold}"/"metrics.csv")
            for stage in [1, 2, 3, 4]:
                fold_df[f"Stage {stage}"].append(100*df[f"acc_{stage}"].loc[0])
fold_df = pd.DataFrame(fold_df)

fold_df[["Evaluation Accuracy", "Training Accuracy"]].plot.box()
plt.title("Evaluation vs. Training Top 1 Accuracy in 5-Fold CV")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid()
# plt.show()
plt.savefig(base_path/"eval_train_fold_accuracy.png")
plt.close()

fold_df[[f"Stage {stage}" for stage in [1, 2, 3, 4]]].plot.box()
plt.title("Multiclass Accuracy Across 5-Fold CV")
plt.ylabel("Accuracy (%)")
plt.xlabel("Pressure Injury Stage (Severity)")
plt.ylim(0, 100)
plt.grid()
# plt.show()
plt.savefig(base_path/"multiclass_fold_accuracy.png")
