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

| 模型 | 特徵 / 表示 | Accuracy | Macro-F1 |
|------|-----------|----------|----------|
| 邏輯回歸 | TF-IDF | 0.76 | 0.75 |
| 線性 SVM | TF-IDF | 0.76 | 0.75 |
| BiLSTM | word2vec（自訓） | 0.54 | 0.55 |
| **BERT（chinese-roberta-wwm-ext）** | 預訓練語言模型微調 | **0.83** | **0.82** |

混淆矩陣：`confusion_baseline.png`（baseline）、`confusion_bilstm.png`、`confusion_bert.png`（Colab 產出）。

### 觀察
- **預訓練語言模型 BERT（83%）表現最佳，明顯優於 TF-IDF（76%）與自訓 word2vec+BiLSTM（54%）。**
- 三種做法的對比剛好說明「表示學習」的演進:
  - **自訓 word2vec + BiLSTM（54%）最弱**:資料僅 567 篇,詞向量從零訓練語料太少、品質有限,深度模型也容易過擬合。
  - **TF-IDF + 線性分類器（76%）是強而穩的 baseline**:新聞主題高度依賴關鍵詞（財經有「股市」、體育有「球員」），簡單特徵就很有效。
  - **BERT（83%）勝出**:借助大規模語料的預訓練知識,即使下游資料量小,微調後仍能大幅領先。
- per-class 觀察:娛樂（F1 0.98）、體育（0.95）最易辨識;**財經 recall 僅 0.53**,常被誤判為科技／國際等相近主題（見混淆矩陣）。
- **結論**:資料量小的中文主題分類任務上,**預訓練模型（BERT）> 傳統特徵（TF-IDF）> 自訓詞向量深度模型**,印證預訓練對小資料任務的價值。
