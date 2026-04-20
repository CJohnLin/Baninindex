# Banini AI: 台股社群反指標狙擊系統 (Local GPU Edition)

這是一個結合自動化爬蟲、自然語言處理 (NLP) 與深度學習的端到端框架。最初基於「巴逆逆（8zz）」的反指標傳說，經過全面升級後，現在已能成為具備**時間排程自動化**與**多維度情緒分析**的 Telegram 零成本專屬交易前瞻預警系統。

## 🌟 核心特色
- **0 成本部署**：捨棄昂貴的雲端 LLM API (如 OpenAI) 以及付費爬蟲，純粹透過本地端強大的 CUDA 算力 (RTX 3090 等 GPU) 驅動。
- **神經網路分析**：使用經過 Fine-tuned 的 `bert-base-chinese` 預訓練模型，深度剖析社群股友的「崩潰程度」與「自信指標」。
- **自動化戰報推播**：內建時程系統 (`JobQueue`)，於台股每日關鍵時間 (13:00 收盤前、13:45 結算後) 準時遞送反向操作情報至您的手機 Telegram。

---

## 🏗️ 系統架構流 (Data Pipeline)

這個專案由四個關鍵的獨立模組構成：

### 1. 無頭瀏覽器深潛爬蟲 (`scrape_threads.py`)
使用非同步的 `playwright` 與 GraphQL 攔截技術，無情突破 Threads 的頁面限制，將指定帳號的歷史財經發言完整結構化。

### 2. 資料清洗與自動打標 (`process_data.py` & `align_market_data.py`)
- 自動過濾雜亂的網路用語、超連結。
- 連接 `yfinance`，把「貼文發布當下」與「未來三天台股實際走勢」的收盤數據完美配對。產生具有黃金價值的訓練用 Ground Truth。

### 3. 本地端模型煉丹 (`train_model.py`)
一鍵即可載入清洗好的樣本資料集，使用 PyTorch 的 AdamW 與您的 NVIDIA 顯示卡，自我訓練並產生最適合您策略的權重檔 `models/banini_model.pt`。

### 4. Telegram 指揮中心 (`telegram_bot.py`)
部署在本地主機常駐的控制器：
- `/banini`：手動即時深潛分析最新動態。
- `/subscribe`：訂閱服務，機器人將於盤中、盤尾自動爬取資料並經過模型判斷後，若有高反轉行情，將發送警報推送。

---

## 🛠️ 環境配置與啟動指南

### 1. 安裝本地依賴
請確保您已安裝具備 CUDA 支援的 PyTorch 環境，然後執行以下指令：
```powershell
# 安裝大數據、機器學習與爬蟲必要套件
pip install playwright parsel nested-lookup jmespath yfinance pandas torch transformers scikit-learn tqdm python-telegram-bot

# 部署無頭瀏覽器環境
python -m playwright install chromium
```

### 2. 模型訓練與資料建置 (可選)
如果您想自建更龐大的資料庫與重訓演算法：
```powershell
python scripts/collect_huge_data.py     # 深度收集歷史貼文
python scripts/process_data.py          # 文本正規清洗
python scripts/align_market_data.py     # 對齊 YFinance 市場數據生成標籤
python scripts/train_model.py           # 進入 GPU 迭代訓練模型
```

### 3. 一鍵啟動 AI 雷達 (Telegram)
請在 `scripts/telegram_bot.py` 的第九行輸入您從 `@BotFather` 取得的 Token，接著直接啟動：
```powershell
python scripts/telegram_bot.py
```
> **Tip:** 此終端機視窗只要不關閉，您的雷達系統將全天候防護！

---

## 🎭 報告維度解讀指南
您從 Telegram 收到的 AI 報表將會包含：
- 🏷️ **產業類別**：自動標記資金意向 (如：半導體、航運、期權)。
- 🎭 **情緒劇本**：辨識出「自信得意 (見頂預測)」或「含淚出場 (見底止跌)」。
- 🎯 **冥燈指數**：由模型根據您的資料庫綜合出來的反指標勝率 (超過 80% 即觸發警報)。

---
*Disclaimer：此開源模型與抓取分析資料僅供研究與技術交流，預測模型均基於過去機率分布演算。專案不對任何投資決策與投資損益負起連帶責任。*
