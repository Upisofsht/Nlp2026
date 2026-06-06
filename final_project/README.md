# 中文新聞主題多分類（NLP 期末專案）

自爬中央社 CNA 多分類新聞，比較 **TF-IDF + 線性分類器**、**word2vec + BiLSTM** 與 **BERT 微調** 三種做法。

## 任務
- 資料：自爬中央社 CNA 10 大分類新聞（政治／國際／兩岸／財經／科技／生活／社會／文化／體育／娛樂），共 973 篇。
- 標籤：以網站分類為標籤（自動標註）。
- 模型：傳統特徵 baseline vs 深度模型，比較主題分類效果。

## 安裝
```
pip install -r requirements.txt
```

## 流程
1. `python scripts/run_crawl.py 200 0.5` — 爬蟲產生 `data/news_dataset.csv`
2. `python scripts/run_baseline.py`      — TF-IDF baseline（logreg / svm）
3. `notebooks/final_project.ipynb`       — 在 Colab（GPU）跑 word2vec + BiLSTM

## 測試
```
pytest -v
```

## 專案結構
- `src/crawler.py` — CNA API 爬蟲與文章解析
- `src/preprocess.py` — jieba 斷詞、清理、停用詞
- `src/features.py` — TF-IDF、word2vec/fastText、序列轉換
- `src/models.py` — 線性分類器 baseline、BiLSTM
- `src/evaluate.py` — accuracy / Macro-F1 / 混淆矩陣

## 結果

資料集：973 篇，10 類，train/test = 80/20（測試 195 篇）。

| 模型 | 特徵 / 表示 | Accuracy | Macro-F1 |
|------|-----------|----------|----------|
| 邏輯回歸 | TF-IDF | 0.70 | 0.68 |
| 線性 SVM | TF-IDF | 0.69 | 0.69 |
| BiLSTM | word2vec（自訓） | （用 10 類資料重跑後填入） | |
| BERT（chinese-roberta-wwm-ext） | 預訓練語言模型微調 | （用 10 類資料重跑後填入） | |

混淆矩陣：`confusion_baseline.png`（baseline）、`confusion_bilstm.png`、`confusion_bert.png`（Colab 產出）。

> 註：6 類版本的舊結果為 TF-IDF 0.76、BiLSTM 0.54、BERT 0.83。改為 10 類後類別更多、部分主題相近（兩岸 vs 政治、社會 vs 地方），準確率自然略降。

### 觀察
- **TF-IDF baseline 在 10 類上約 70% / Macro-F1 0.68–0.69**，仍是強而穩的 baseline：新聞主題高度依賴關鍵詞，簡單特徵就很有效。
- 類別數從 6 增到 10 後準確率下降，主因是新增類別與既有類別語意相近（兩岸／政治、社會／生活），較易互相誤判。
- BiLSTM 與 BERT 需用 10 類資料重跑後再填入；預期排序仍為 **BERT > TF-IDF > 自訓 word2vec+BiLSTM**。
