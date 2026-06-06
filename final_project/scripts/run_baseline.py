import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import tokenize, load_stopwords
from src.features import build_tfidf
from src.models import train_baseline
from src.evaluate import evaluate_predictions, plot_confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "news_dataset.csv")
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna(""))
    stop = load_stopwords(ROOT / "stopwords.txt")
    df["tokens"] = df["text"].apply(lambda t: " ".join(tokenize(t, stop)))

    X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
        df["tokens"], df["category"], test_size=0.2, stratify=df["category"], random_state=42)

    X_tr, vec = build_tfidf(X_tr_txt)
    X_te = vec.transform(X_te_txt)

    for kind in ["logreg", "svm"]:
        clf = train_baseline(X_tr, y_tr, kind=kind)
        pred = clf.predict(X_te)
        m = evaluate_predictions(y_te, pred)
        print(f"\n=== {kind} (TF-IDF) ===")
        print(f"accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
        print(m["report"])

    labels = sorted(df["category"].unique())
    plot_confusion_matrix(y_te, pred, labels, out_path=str(ROOT / "confusion_baseline.png"))
    print("混淆矩陣已存 confusion_baseline.png")
