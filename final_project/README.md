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
| BiLSTM | word2vec | （Colab 執行後填入） | （Colab 執行後填入） |

混淆矩陣：`confusion_baseline.png`（baseline）、`confusion_bilstm.png`（Colab 產出）。

### 觀察
- TF-IDF + 線性分類器在新聞主題分類上即可達到 ~76% accuracy，因各類別關鍵詞差異明顯。
- BiLSTM + 詞嵌入結果待 Colab 訓練後補上，並與 baseline 對比。
