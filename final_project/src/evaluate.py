from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred),
    }


def _use_cjk_font():
    """讓 matplotlib 優先用支援中文的字型,找不到就 fallback(不會報錯)。"""
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC", "Noto Sans CJK JP", "Microsoft JhengHei",
        "PingFang TC", "Heiti TC", "SimHei", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(y_true, y_pred, labels, out_path="confusion_matrix.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    _use_cjk_font()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path
