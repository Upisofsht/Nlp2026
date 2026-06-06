import re
import jieba

# 移除 HTML tag、標點符號，保留中英數與空白
_HTML_RE = re.compile(r"<[^>]+>")
_PUNCT_RE = re.compile(r"[^一-鿿A-Za-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _HTML_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def load_stopwords(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    stopwords = stopwords or set()
    cleaned = clean_text(text)
    tokens = jieba.lcut(cleaned)
    return [t for t in tokens if t.strip() and t not in stopwords]
