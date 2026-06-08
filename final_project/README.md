# 中文新聞主題多分類（NLP 期末專案）

自爬中央社 CNA 多分類新聞，比較 **TF-IDF + 邏輯回歸（傳統機器學習）** 與 **BERT 微調（預訓練語言模型）** 兩種做法。

## 任務
- 資料：自爬中央社 CNA 10 大分類新聞（政治／國際／兩岸／財經／科技／生活／社會／文化／體育／娛樂），共 973 篇。
- 標籤：以網站分類為標籤（自動標註）。
- 模型：傳統特徵 baseline vs 預訓練語言模型，比較主題分類效果。

## 在 Colab 執行（建議）
整個專案可在單一 notebook 從頭跑到尾：

**`notebooks/final_project.ipynb`** — 開 GPU 後依序執行：
安裝套件 → clone repo → 載入資料 → jieba 前處理 → 切分 → TF-IDF → BERT 微調 → 結果比較。
預設使用 repo 內附的 `data/news_dataset.csv`（973 篇），結果可重現；爬蟲為可選示範。

## 在本機執行
```
pip install -r requirements.txt
python scripts/run_crawl.py 200 0.5   # （可選）爬蟲產生 data/news_dataset.csv
python scripts/run_baseline.py        # TF-IDF baseline（logreg / svm）
```

## 測試
```
pytest -v
```

## 專案結構
- `src/crawler.py` — CNA API 爬蟲與文章解析
- `src/preprocess.py` — jieba 斷詞、清理、停用詞
- `src/features.py` — TF-IDF 特徵
- `src/models.py` — 線性分類器 baseline
- `src/evaluate.py` — accuracy / Macro-F1 / 混淆矩陣
- `notebooks/final_project.ipynb` — Colab 一條龍：TF-IDF + BERT 微調與比較

## 結果

資料集：973 篇，10 類，train/test = 80/20（測試 195 篇）。

| 模型 | 特徵 / 表示 | Accuracy | Macro-F1 |
|------|-----------|----------|----------|
| TF-IDF + 邏輯回歸 | 關鍵詞頻率 | 0.70 | 0.68 |
| **BERT（chinese-roberta-wwm-ext）** | 預訓練語言模型微調 | **0.77** | **0.73** |

混淆矩陣：`confusion_baseline.png`（TF-IDF）、`confusion_bert.png`（BERT，Colab 產出）。

### 觀察
- **BERT（77%）明顯優於 TF-IDF baseline（70%）**：預訓練語言模型即使在 10 類、資料量仍有限的情況下，依然領先。
- **TF-IDF + 邏輯回歸（~70%）是強而穩的 baseline**：新聞主題高度依賴關鍵詞，簡單特徵就很有效、訓練快、可解釋。
- **「財經」類最難，即使 BERT 的 F1 也只有 0.11**：18 篇測試樣本只認出 1 篇（recall 0.06），其餘散落到生活、體育、政治、文化等類；但模型一旦預測為財經幾乎都對（precision 1.00），顯示它對財經過度保守。這是類別語意重疊造成的典型混淆，也是本實驗最值得討論的現象（見混淆矩陣）。
- per-class（BERT）：社會 0.89、文化 0.86、體育 0.86 最易辨識；財經 0.11 最難。
