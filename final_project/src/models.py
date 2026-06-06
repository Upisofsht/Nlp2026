from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_baseline(X_train, y_train, kind="logreg"):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif kind == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(f"unknown kind: {kind}")
    clf.fit(X_train, y_train)
    return clf
