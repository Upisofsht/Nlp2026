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
| BiLSTM | word2vec（自訓） | 0.52 | 0.50 |
| **BERT（chinese-roberta-wwm-ext）** | 預訓練語言模型微調 | **0.77** | **0.73** |

混淆矩陣：`confusion_baseline.png`（baseline）、`confusion_bilstm.png`、`confusion_bert.png`（Colab 產出）。

> 註：6 類版本的舊結果為 TF-IDF 0.76、BiLSTM 0.54、BERT 0.83。改為 10 類後類別更多、部分主題相近（兩岸 vs 政治、社會 vs 生活），準確率自然略降。

### 觀察
- **三種做法排序：BERT（77%）> TF-IDF（70%）> 自訓 word2vec+BiLSTM（52%）。** 預訓練語言模型即使在 10 類、資料量仍有限的情況下,依然明顯領先。
- **TF-IDF + 線性分類器（~70%）是強而穩的 baseline**：新聞主題高度依賴關鍵詞,簡單特徵就很有效;自訓 word2vec 因語料小、詞向量品質有限,BiLSTM 反而最弱。
- **「財經」類在 BiLSTM 與 BERT 皆為 F1=0**：模型幾乎不曾預測出財經,全被歸入科技／政治／兩岸等語意相近的類別——這是類別重疊造成的典型混淆,也是本實驗最值得討論的現象（見混淆矩陣）。
- 類別數從 6 增到 10 後整體準確率下降,主因是新增類別與既有類別語意相近（兩岸／政治、社會／生活）。
- per-class（BERT）：娛樂 0.90、兩岸 0.88、體育 0.88、文化 0.86 最易辨識;財經 0.00 最難。
