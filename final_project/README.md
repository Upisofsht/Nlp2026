# 中文新聞主題多分類（NLP 期末專案）

自爬中央社 CNA 多分類新聞，比較 **TF-IDF + 線性分類器** 與 **word2vec + BiLSTM** 兩種做法。

## 任務
- 資料：自爬中央社 CNA 6 大分類新聞（政治／國際／財經／科技／體育／娛樂），共 567 篇。
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

資料集：567 篇，6 類，train/test = 80/20。

| 模型 | 特徵 | Accuracy | Macro-F1 |
|------|------|----------|----------|
| 邏輯回歸 | TF-IDF | 0.76 | 0.75 |
| 線性 SVM | TF-IDF | 0.76 | 0.75 |
| BiLSTM | word2vec | 0.54 | 0.55 |

混淆矩陣：`confusion_baseline.png`（baseline）、`confusion_bilstm.png`（Colab 產出）。

### 觀察
- **TF-IDF + 線性分類器（76%）明顯優於 BiLSTM（54%）。** 這在新聞主題分類上是合理且常見的結果：
  - 主題分類高度依賴「關鍵詞」（財經有「股市」、體育有「球員」），TF-IDF 直接捕捉這些區辨性詞彙。
  - 本資料集僅 567 篇（訓練約 450 篇），word2vec 從零訓練、語料太小，詞向量品質有限；BiLSTM 參數多，小資料容易欠擬合/過擬合。
  - 深度序列模型（BiLSTM）的優勢通常要在「大量資料」或「語意/語序敏感」任務上才會顯現。
- **結論**：在小型主題分類任務上，傳統特徵反而更強、更穩——這正是本專案的重點發現。
- 可能的改進方向（未來工作）：擴大資料量、改用預訓練中文詞向量（騰訊／fastText 官方）、增加訓練 epoch、加強正則化，或改用 BERT 微調。
