import pathlib
from src.crawler import parse_listing, parse_article

FIX = pathlib.Path(__file__).parent / "fixtures"


def _read(name):
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_listing_extracts_article_links():
    links = parse_listing(_read("listing_sample.html"), base="https://www.cna.com.tw")
    assert "https://www.cna.com.tw/news/aspt/202606060001.aspx" in links
    assert len(links) == 2


def test_parse_article_extracts_title_and_content():
    art = parse_article(_read("article_sample.html"))
    assert art["title"] == "台積電法說會登場"
    assert "第一段內文。" in art["content"]
    assert "第二段內文。" in art["content"]
