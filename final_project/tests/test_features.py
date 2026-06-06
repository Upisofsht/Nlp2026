from src.features import build_tfidf


def test_build_tfidf_shapes():
    docs = ["台積電 股價 大漲", "球員 進球 得分", "台積電 法說會"]
    X, vec = build_tfidf(docs)
    assert X.shape[0] == 3
    assert X.shape[1] == len(vec.get_feature_names_out())
    assert X.shape[1] > 0


from src.features import train_word2vec, build_vocab, texts_to_sequences


def test_train_word2vec_and_sequences():
    tokenized = [["台積電", "股價", "大漲"], ["球員", "進球", "得分"], ["台積電", "法說會"]]
    w2v = train_word2vec(tokenized, vector_size=20, min_count=1, epochs=5)
    assert w2v.wv["台積電"].shape == (20,)

    word2idx = build_vocab(tokenized)
    assert word2idx["<PAD>"] == 0
    seqs = texts_to_sequences(tokenized, word2idx, maxlen=4)
    assert seqs.shape == (3, 4)
