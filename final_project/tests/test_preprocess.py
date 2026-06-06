from src.preprocess import clean_text, tokenize


def test_clean_text_removes_html_and_punct():
    raw = "<p>台積電股價大漲！！</p>  3000 元"
    assert clean_text(raw) == "台積電股價大漲 3000 元"


def test_tokenize_returns_tokens_without_stopwords():
    tokens = tokenize("台積電的股價大漲", stopwords={"的"})
    assert "的" not in tokens
    assert "台積電" in tokens
    assert all(isinstance(t, str) for t in tokens)


def test_tokenize_filters_whitespace_tokens():
    tokens = tokenize("台積電  大漲", stopwords=set())
    assert "" not in tokens
    assert " " not in tokens
