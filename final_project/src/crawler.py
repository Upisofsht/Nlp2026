import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (NLP course project crawler)"}

# CNA 分類代碼 → 標籤（實作時依實際頁面確認/調整）
CATEGORIES = {
    "aipl": "政治",
    "aopl": "國際",
    "afe": "財經",
    "ait": "科技",
    "aspt": "體育",
    "amov": "娛樂",
}


def parse_listing(html: str, base: str = "https://www.cna.com.tw") -> list:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("ul.mainList li a"):
        href = a.get("href")
        if href:
            links.append(urljoin(base, href))
    return links


def parse_article(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("h1.centralContent")
    title = title_el.get_text(strip=True) if title_el else ""
    paras = soup.select("div.paragraph p")
    content = " ".join(p.get_text(strip=True) for p in paras)
    return {"title": title, "content": content}


def fetch(url: str, sleep: float = 1.0) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    time.sleep(sleep)  # 禮貌爬
    return resp.text
