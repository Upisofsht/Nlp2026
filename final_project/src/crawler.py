import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (NLP course project crawler)"}
API_URL = "https://www.cna.com.tw/cna2018api/api/WNewsList"

# CNA 分類代碼 → 標籤
CATEGORIES = {
    "aipl": "政治",
    "aopl": "國際",
    "acn": "兩岸",
    "afe": "財經",
    "ait": "科技",
    "ahel": "生活",
    "asoc": "社會",
    "acul": "文化",
    "aspt": "體育",
    "amov": "娛樂",
}


def parse_api_items(data: dict) -> list[str]:
    """從 CNA WNewsList API 的 JSON 取出文章網址清單。"""
    result_data = data.get("ResultData") or {}
    items = result_data.get("Items") or []
    return [it["PageUrl"] for it in items if it.get("PageUrl")]


def parse_article(html: str) -> dict:
    """解析單篇文章頁，回傳 {title, content}。"""
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("div.centralContent h1")
    title = title_el.get_text(strip=True) if title_el else ""
    paras = soup.select("div.paragraph p")
    content = " ".join(p.get_text(strip=True) for p in paras)
    return {"title": title, "content": content}


def fetch(url: str, sleep: float = 1.0) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(sleep)  # 禮貌爬
    return resp.text


def list_article_urls(category: str, n_articles: int = 100,
                      pagesize: int = 20, sleep: float = 1.0) -> list[str]:
    """用 CNA API 分頁列出某分類的文章網址，直到湊滿 n_articles 或沒有下一頁。"""
    urls: list[str] = []
    pageidx = 1
    while len(urls) < n_articles:
        payload = {"action": "0", "category": category,
                   "pagesize": str(pagesize), "pageidx": pageidx}
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        page_urls = parse_api_items(data)
        if not page_urls:
            break
        urls.extend(page_urls)
        time.sleep(sleep)
        next_idx = (data.get("ResultData") or {}).get("NextPageIdx")
        if not next_idx:
            break
        pageidx = next_idx
    return urls[:n_articles]


def crawl(categories: dict = None, n_per_category: int = 100,
          sleep: float = 1.0) -> list[dict]:
    """回傳 [{title, content, category, url}, ...]"""
    categories = categories or CATEGORIES
    rows = []
    for code, label in categories.items():
        try:
            urls = list_article_urls(code, n_per_category, sleep=sleep)
        except Exception as e:
            print(f"[warn] 分類 {label} 列表抓取失敗: {e}")
            continue
        for url in urls:
            try:
                art = parse_article(fetch(url, sleep))
            except Exception as e:
                print(f"[warn] 文章抓取失敗 {url}: {e}")
                continue
            if art["title"] and art["content"]:
                rows.append({**art, "category": label, "url": url})
        print(f"[ok] {label}: 累計 {len(rows)} 篇")
    return rows
