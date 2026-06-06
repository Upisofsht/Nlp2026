from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(docs, max_features=20000):
    """docs: 已用空白接好的斷詞字串清單。回傳 (稀疏矩陣, vectorizer)。"""
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)
    return X, vec
