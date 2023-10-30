import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from pathlib import Path


class Metrics:
    """
    Metrics to calculate: accuracy, macro f1, auc, sensitivity, specificity, tpr, fpr
    """

    def __init__(self, model_name, labels):
        self.model_name = model_name
        self.base_path = f"model_metrics/{model_name}"
        Path(self.base_path).mkdir(exist_ok=True, parents=True)
        self.df = {"Model": [model_name], "Accuracy": [], "F1-Score": [], "Sensitivity": [],
                   "Specificity": [], "TPR": [], "FPR": [], "PPV": [], "NPV": [], "AUC": [], "acc_1": [], "acc_2": [], "acc_3": [],
                   "acc_4": []}
        self.labels = labels
        self.classes = len(self.labels)

    def evaluate(self, truths, probabilities):
        for i, row in enumerate(probabilities):
            row = [((j-min(row)) / (max(row) - min(row)))/4 for j in row]
            row[0] -= 1-sum(row)
            probabilities[i] = row

        predictions = np.array(probabilities).argmax(axis=1)
        # print(truths, predictions)
        # summary statistics
        accuracy = sklearn.metrics.accuracy_score(truths, predictions)
        f1 = sklearn.metrics.f1_score(truths, predictions, average="macro")

        matrix = sklearn.metrics.confusion_matrix(truths, predictions)
        per_class_accuracy = (matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]).diagonal()
        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix.sum() - (FP + FN + TP)
        sensitivities = TP / (TP + FN)
        sensitivity = np.mean(sensitivities)
        specificities = TN / (TN + FP)
        specificity = np.mean(specificities)
        tpr = np.mean(TP / (TP + FN))
        fpr = np.mean(FP / (FP + FN))
        ppv = np.mean(TP / (TP + FP))
        npv = np.mean(TN / (TN + FN))
        print(probabilities)
        auc = sklearn.metrics.roc_auc_score(
            truths,
            probabilities,
            multi_class="ovr",
            average="micro",
        )

        self.df["Accuracy"].append(accuracy)
        self.df["F1-Score"].append(f1)
        self.df["Specificity"].append(specificity)
        self.df["Sensitivity"].append(sensitivity)
        self.df["TPR"].append(tpr)
        self.df["FPR"].append(fpr)
        self.df["PPV"].append(ppv)
        self.df["NPV"].append(npv)
        self.df["AUC"].append(auc)
        for i, acc in enumerate(per_class_accuracy):
            col = f"acc_{i+1}"
            self.df[col].append(acc)
        print(self.df)
        self.create_metrics_df()

    def roc_auc(self, truths, probabilities):
        auc = sklearn.metrics.roc_auc_score(truths, probabilities, multi_class="ovr")
        # ROC curve
        fig, ax = plt.subplots(figsize=(6, 6))
        truths = label_binarize(truths, classes=[*range(self.classes)])
        truths = np.array(truths)
        probabilities = np.array(probabilities)
        for class_id, color in zip([*range(self.classes)], ["red", "blue", "green", "purple"]):
            sklearn.metrics.RocCurveDisplay.from_predictions(
                truths[:, class_id],
                probabilities[:, class_id],
                name=f"Severity {self.labels[class_id]} vs all other classes",
                color=color,
                ax=ax
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves for {self.model_name}")
        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.savefig(f"{self.base_path}/roc_curve.png")
        plt.close()

    def create_metrics_df(self):
        filepath = f"{self.base_path}/metrics.csv"
        self.df = pd.DataFrame(self.df)
        self.df.to_csv(filepath, index=False)
        dfs = []
        for path in Path("model_metrics").iterdir():
            if path.is_dir() and (path / "metrics.csv").exists():
                metrics_path = str(path / "metrics.csv")
                dfs.append(pd.read_csv(metrics_path))
        total_df = pd.concat(dfs)
        total_df.to_csv("model_metrics/metrics.csv", index=False)
        return total_df

    def plot_individual_model_metrics(self, history):
        acc = history['train_acc']
        val_acc = history['val_acc']
        loss = history['train_loss']
        val_loss = history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylim([0, 5])
        plt.savefig(f"{self.base_path}/metrics_plot.png")
        plt.close()

    @staticmethod
    def plot_all_model_metrics(df):
        df = df.set_index("Model")
        df = df.T
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        df.plot.bar(ax=ax)
        box = ax.get_position()
        fig.suptitle("Metrics Comparison of All Models")
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_ylim(0, 1)

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("model_metrics/bar_plot.png")
        plt.close()
