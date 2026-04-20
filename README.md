# Banini AI: 全自動台股反指標狙擊系統 (Local Evolution Edition)

這是一個具備 **「自我進化能力」** 的端到端財經 AI 系統。它能在您的本地電腦 24/7 運作，自動爬取社群指標性帳號（如：巴逆逆），透過 BERT 預測市場反轉勝率，並根據真實股市結果進行自我迭代。

## 🌟 核心特點 (Core Pillars)

- **🤖 持續學習 (Continuous Learning)**：系統具備自動化打標與重訓機制。
    - **觀察**：每日盤中自動記錄 AI 的預測值。
    - **檢討**：3 天後自動串接 `yfinance` 檢查預測是否準確，並存入黃金資料庫。
    - **進化**：每週日凌晨 03:00 自動背景重訓模型並執行「熱重載」，大腦每週更新一次。
- **🔌 0 成本、高安全**：完全運作於本地 NVIDIA GPU (CUDA)，不依賴付費 LLM API。您的數據與策略完全私有化。
- **📊 實戰級分析維度**：
    - **產業標籤**：自動識別貼文狙擊的板塊（半導體、航運、ETF 等）。
    - **情緒判別**：辨別「極度崩潰 (反向看多)」或「自信爆棚 (反向看空)」。
- **📱 全功能 Telegram 指揮中心**：支援主動警報推送、歷史勝率排行、自訂情境模擬。

---

## 🏛️ 系統指令集 (Telegram Commands)

- `/start`：啟動系統並顯示選單。
- `/banini`：手動觸發最新情境深度掃描。
- `/manual <標的>`：模擬分析。*（例：若她現在看多 2330，AI 會給出多高的反向風險？）*
- `/rank`：歷史戰績榜。統計哪檔股票被點名後的「冥燈勝率」最高。
- `/sentiment`：量測市場情緒溫度計。同時掃描多個財經帳號做綜合評分。
- `/subscribe [門檻]`：訂閱自動定時推播。
- `/set_alert <門檻>`：客製化警報。例如設定 `0.8` 以上才發送緊急通知。

---

## 🏗️ 數據流水線 (Pipeline)

1. **爬蟲層 (`scrape_threads.py`)**: 使用 Playwright + GraphQL 攔截技術。
2. **自動驗證層 (`auto_labeler.py`)**: 每日凌晨自動回測 3 天前的發文與股價報酬率。
3. **訓練引擎 (`train_model.py`)**: 基於 BERT 的分類模型，使用 AdamW 與 GPU 加速訓練。
4. **控制中心 (`telegram_bot.py`)**: 內建 JobQueue 非同步事件迴圈，管理所有排程任務。

---

## 🛠️ 環境配置

```powershell
# 安裝機器學習與爬蟲必要套件
pip install playwright pandas yfinance torch transformers apscheduler python-telegram-bot scikit-learn tqdm

# 部署無頭瀏覽器環境
python -m playwright install chromium

# 啟動系統 (只要啟動 Bot，排程就會自動運行)
python scripts/telegram_bot.py
```

> **Tip:** 為確保進化機制正常運作，建議每週日凌晨保持電腦開啟，以利 GPU 執行神經網路權重更新。

---
*Disclaimer：此專案僅供技術研究與量化模型開發交流使用，預測結果並非投資建議。交易風險請自行負擔。*
