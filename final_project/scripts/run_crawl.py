import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src.crawler import crawl

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    sleep = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    rows = crawl(n_per_category=n, sleep=sleep)
    df = pd.DataFrame(rows)
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "news_dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n完成：{len(df)} 篇，各類分布：")
    print(df["category"].value_counts())
    print(f"已存：{out}")
