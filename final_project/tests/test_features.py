from src.features import build_tfidf


def test_build_tfidf_shapes():
    docs = ["台積電 股價 大漲", "球員 進球 得分", "台積電 法說會"]
    X, vec = build_tfidf(docs)
    assert X.shape[0] == 3
    assert X.shape[1] == len(vec.get_feature_names_out())
    assert X.shape[1] > 0
