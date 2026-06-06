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


def build_bilstm(vocab_size, num_classes, embedding_matrix=None,
                 vector_size=100, maxlen=200):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, Bidirectional, LSTM,
                                          Dense, Dropout)

    emb_kwargs = dict(input_dim=vocab_size, output_dim=vector_size, input_length=maxlen)
    if embedding_matrix is not None:
        emb_kwargs.update(weights=[embedding_matrix], trainable=True)

    model = Sequential([
        Embedding(**emb_kwargs),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
