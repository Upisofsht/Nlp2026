from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec, FastText


def build_tfidf(docs, max_features=20000):
    """docs: 已用空白接好的斷詞字串清單。回傳 (稀疏矩陣, vectorizer)。"""
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)
    return X, vec


def train_word2vec(tokenized_docs, vector_size=100, min_count=2, epochs=10, use_fasttext=False):
    Model = FastText if use_fasttext else Word2Vec
    model = Model(sentences=tokenized_docs, vector_size=vector_size,
                  window=5, min_count=min_count, workers=4, epochs=epochs)
    return model


def build_vocab(tokenized_docs, max_vocab=30000):
    from collections import Counter
    counter = Counter(tok for doc in tokenized_docs for tok in doc)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab):
        word2idx[word] = len(word2idx)
    return word2idx


def texts_to_sequences(tokenized_docs, word2idx, maxlen=200):
    seqs = np.zeros((len(tokenized_docs), maxlen), dtype="int32")
    for i, doc in enumerate(tokenized_docs):
        for j, tok in enumerate(doc[:maxlen]):
            seqs[i, j] = word2idx.get(tok, word2idx["<UNK>"])
    return seqs


def build_embedding_matrix(word2idx, w2v_model, vector_size=100):
    matrix = np.zeros((len(word2idx), vector_size), dtype="float32")
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
    return matrix
