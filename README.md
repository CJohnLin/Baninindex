# Banini AI: 全自動台股反指標狙擊系統 (Local Evolution Edition)

這是一個具備 **「自我進化能力」** 的端到端財經 AI 系統。它能在您的本地電腦 24/7 運作，自動爬取社群指標性帳號（如：巴逆逆），透過 BERT 預測市場反轉勝率，並根據真實股市結果進行自我迭代。

## 🌟 核心特點 (Core Pillars)

- **🤖 持續學習 (Continuous Learning)**：系統具備自動化打標與重訓機制。
    - **觀察**：每日盤中自動記錄 AI 的預測值。
    - **檢討**：3 天後自動串接 `yfinance` 檢查預測是否準確，並存入黃金資料庫。
    - **進化**：每週日凌晨 03:00 自動背景重訓模型並執行「熱重載」，大腦每週更新一次。
- **🔌 0 成本、超輕量**：完全運作於本地 NVIDIA GPU (CUDA)，不依賴付費 LLM API。搭載 **Lazy Load (延遲載入)** 架構，閒置時不佔用 VRAM。
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

## 🏗️ 數據流水線與架構 (Pipeline & Architecture)

本系統採 **高度解耦 (Layered & Decoupled)** 架構，確保資源的高效利用與靈活性：

1. **獨立大腦層 (`agent_core.py`)**: 本系統的 AI 核心，負責統籌爬蟲與 BERT 模型推論。
    - **Lazy Load (延遲載入)**：AI 模型只在指令觸發時動態掛載至 GPU，閒置時不霸佔 VRAM。
    - **Standalone 執行**：無須設定 `.env` 或 Telegram Token，直接跑 `python scripts/agent_core.py` 即可獲得分析報告。
2. **無頭爬蟲層 (`scrape_threads.py`)**: 透過 Playwright 靜默攔截 API 提取數據，無須付費金鑰。
3. **自我進化層 (`auto_labeler.py`, `train_model.py`)**: 每日對 3 天前的預測自動打標回測，並據此重訓大腦權重。
4. **展示與控制中心 (`telegram_bot.py`)**: 純粹的介面層（Layer 3），只負責與使用者對話並管理排程，將運算重擔交給底層 Core 即時執行。

---

## 🛠️ 環境配置

```powershell
# 安裝機器學習與爬蟲必要套件
pip install playwright pandas yfinance torch transformers apscheduler python-telegram-bot scikit-learn tqdm

# 部署無頭瀏覽器環境
python -m playwright install chromium

# 方式 A：免配置、純 CLI 本地測試
python scripts/agent_core.py

# 方式 B：啟動完整 Telegram 指揮中心 (需設定 .env)
python scripts/telegram_bot.py
```

> **Tip:** 為確保進化機制正常運作，建議每週日凌晨保持電腦開啟，以利 GPU 執行神經網路權重更新。

---
*Disclaimer：此專案僅供技術研究與量化模型開發交流使用，預測結果並非投資建議。交易風險請自行負擔。*
