import json
import pathlib
from src.crawler import parse_api_items, parse_article

FIX = pathlib.Path(__file__).parent / "fixtures"


def _read(name):
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_api_items_extracts_article_urls():
    data = json.loads(_read("api_sample.json"))
    urls = parse_api_items(data)
    assert "https://www.cna.com.tw/news/aspt/202606060001.aspx" in urls
    assert len(urls) == 2


def test_parse_api_items_handles_empty():
    assert parse_api_items({}) == []
    assert parse_api_items({"ResultData": {"Items": []}}) == []


def test_parse_article_extracts_title_and_content():
    art = parse_article(_read("article_sample.html"))
    assert art["title"] == "台積電法說會登場"
    assert "第一段內文。" in art["content"]
    assert "第二段內文。" in art["content"]
