# Banini AI: 全自動台股反指標狙擊系統 (Local Evolution & DRL Edition)

## 📝 摘要 (Abstract)

### 🎯 動機 (Motivation)
隨著科技進步與量化交易的發展，社群媒體上的「反指標」（如財經指標性帳號「巴逆逆」）成為市場中另類的參考訊號。如何有系統地捕捉這些非傳統的社群情緒，並將其轉化為自動化的量化交易策略，引起了高度的研究興趣。

### ⚠️ 挑戰 (But)
然而，市場環境瞬息萬變，傳統上單純提供靜態的市場預測或勝率，已不足以應付複雜的交易決策。如何將 AI 的預測準確地轉換為包含資金控管的實際交易動作，同時確保預測模型能夠隨著市場變化持續更新而不失效，是目前面臨的主要挑戰。

### 💊 解決方案 (Cure)
為了解決上述挑戰，我們提出了一個具備 **「自我進化能力」與「深度強化學習 (DRL)」** 的全新端到端財經 AI 系統——**Banini AI**。

### 🏗️ 系統開發 (Development)
我們的方法設計基於自然語言預測與強化學習雙重架構。系統首先透過自動爬蟲跨平台（Threads 與 Facebook）擷取社群發文，利用 BERT 模型評估市場反轉的預測信心指數（BERT Score）。接著，強化學習代理人（Trading Agent）具備持續學習（Continuous Learning）機制，會自動紀錄預測值，於 3 天後與真實股市結果對齊建立「黃金訓練集」，並在每週日自動進行背景重訓與權重更新。

### 📊 實驗評估 (Experiments)
為評估此系統，我們進行了擬真回測與歷史數據擴增（Data Synthesis）。利用內建模組模擬了 2024 年 4 月至今的歷史發文情境進行回測，並串接 `yfinance` 獲取真實市場的 K 線報酬，以此作為獎勵機制（Reward）來訓練並最佳化 Agent 的投資決策。

---

## 🌟 核心特點 (Core Pillars)

- **🧠 強化學習代理人 (DRL Trading Agent)**：系統不再只是提供勝率，還能模擬交易決策。
    - **自主動作**：Agent 會根據預測信心指數（BERT Score）自主決定「重倉」、「輕倉」或「觀望」。
    - **獎勵機制**：串接 `yfinance` 獲取真實市場報酬，以此作為 Reward 訓練 Agent 最佳化投資決策。
- **🤖 持續學習與進化 (Continuous Learning)**：
    - **自動標籤**：每日盤中自動記錄預測值，並在 3 天後自動對齊股市結果，存入「黃金訓練集」。
    - **模型週更**：每週日自動啟動背景重訓與權重更新，確代理人的市場感知與時俱進。
- **📈 數據合成與歷史擴增 (Data Synthesis)**：
    - **擬真回測**：內建 `generate_fb_history.py` 模擬 2024 年 4 月至今的歷史發文情境。
    - **多維報表**：自動生成 `fb_historical_report.md`，統計各標的的「冥燈勝率」排行榜。
- **🔌 0 成本、超輕量**：完全運作於本地 CUDA 加速，閒置時自動釋放 VRAM。支援 Threads 與 Facebook 雙引擎掃描。

---

## 🏛️ 系統指令集 (Telegram Commands)

- `/start`：啟動系統並顯示選單。
- `/banini`：跨平台（Threads & FB）情境深度掃描。
- `/rank`：**歷史戰績榜**。查看哪個標的被點名後的「反指標勝率」最高（如：2317 鴻海）。
- `/sentiment`：量測社群大眾情緒溫度計。
- `/manual <標的>`：針對特定標的進行 AI 模擬分析與動作建議。
- `/subscribe`：訂閱高勝率訊號推播。

---

## 🏗️ 系統架構 (System Architecture)

1. **獨立大腦層 (`agent_core.py`)**: 統籌模型推論與爬蟲調度。
2. **決策代理層 (`trading_agent.py`)**: 負責將 AI 預測轉換為具體的投資動作（Action）。
3. **數據流水線 (`generate_fb_history.py`)**: 負責歷史數據合成與 `yfinance` 真實報酬對齊。
4. **展示與控制中心 (`telegram_bot.py`)**: 負責與使用者對話並展示即時分析報告。

---

## 🛠️ 環境配置與執行

```powershell
# 1. 安裝必要套件
pip install pandas yfinance torch transformers playwright apscheduler python-telegram-bot tqdm

# 2. 初始化爬蟲環境
python -m playwright install chromium

# 3. [可選] 生成歷史擬真數據並訓練 (DRL 資料擴增)
python scripts/generate_fb_history.py
python scripts/train_model.py

# 4. [啟動] 運行 Telegram 指揮中心
python scripts/telegram_bot.py
```

## 📊 歷史分析示例
| 股票代號 | 點名次數 | 反指標勝率 | 3日後平均報酬率 |
| :--- | :--- | :--- | :--- |
| `2317.TW` | 48 次 | **56.2%** | +1.46% |
| `2303.TW` | 70 次 | **54.3%** | -0.24% |

---
> **Disclaimer:** 此專案僅供技術研究與量化開發交流。AI 預測不代表投資建議，任何交易盈虧請自負。

