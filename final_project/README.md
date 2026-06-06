# 中文新聞主題多分類（NLP 期末專案）

自爬中央社 CNA 多分類新聞，比較 TF-IDF+線性分類器 與 word2vec+BiLSTM。

## 安裝
```
pip install -r requirements.txt
```

## 流程
1. `python scripts/run_crawl.py`     — 爬蟲產生 data/news_dataset.csv
2. `python scripts/run_baseline.py`  — TF-IDF baseline
3. `notebooks/final_project.ipynb`   — Colab 跑 word2vec + BiLSTM

## 測試
```
pytest -v
```
