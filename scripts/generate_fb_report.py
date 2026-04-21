import pandas as pd
import os
from datetime import datetime

DATA_FILE = "datasets/processed/aligned_training_data.csv"
OUTPUT_REPORT = "docs/fb_historical_report.md"

def generate_report():
    if not os.path.exists(DATA_FILE):
        print("找不到訓練數據集，無法產生報表。")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # 篩選 FB 模擬歷史資料 (post_id 包含 fb_mock, 或者依照 timestamp)
    fb_df = df[df['post_id'].astype(str).str.contains('fb_mock')]
    
    if fb_df.empty:
        print("未找到 Facebook 的擴增訓練資料。")
        return
        
    total_posts = len(fb_df)
    win_rate = fb_df['is_contrarian_win'].mean()
    
    # 計算冥燈指數 (Contrarian Index)
    # 取決於勝率以及平均報酬率的破壞力
    # 這裡實作一個公式: 勝率 * (1 + 絕對報酬率中位數 * 10) * 10，滿分 10 分
    median_abs_return = fb_df['return_rate'].abs().median()
    contrarian_index = min(10.0, (win_rate * 10) * (1.0 + median_abs_return * 5))
    
    # 依照標的分類統計
    ticker_stats = fb_df.groupby('ticker').agg(
        total_posts=('post_id', 'count'),
        win_rate=('is_contrarian_win', 'mean'),
        avg_return=('return_rate', 'mean')
    ).reset_index()
    
    # 排序：以勝率最高與提及次數最多為優先
    ticker_stats = ticker_stats.sort_values(by=['win_rate', 'total_posts'], ascending=[False, False])
    
    # 產出 Markdown
    md = f"""# 巴逆逆 Facebook 歷史回溯分析報表

**統計區間**: 2024/04/01 起至今
**總結**: 透過歷史股價與擬真 FB 發文事件對齊，產生了真實市場環境下的 RL 擴增訓練集。

## 📊 核心指標 (Core Metrics)

- **總分析貼文數**: {total_posts} 篇
- **反向操作勝率**: **{win_rate:.2%}**
- **冥燈指數 (0-10 分)**: **{contrarian_index:.1f} / 10** (*由勝率與股價實際破壞力綜合評分*)

---

## 🏆 標的勝率排行 (Ticker Leaderboard)

| 股票代號 | 點名次數 | 反指標勝率 | 3日後平均報酬率 |
| :--- | :--- | :--- | :--- |
"""
    for _, row in ticker_stats.iterrows():
        t = row['ticker']
        c = row['total_posts']
        w = row['win_rate']
        r = row['avg_return']
        md += f"| `{t}` | {c} 次 | **{w:.1%}** | {r:+.2%} |\n"
        
    md += """
---
*Disclaimer: 此報告由 DRL Agent 回溯演算法自動生成，為合成歷史評估矩陣。*
"""
    
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"報告已成功生成並儲存至：{OUTPUT_REPORT}")

if __name__ == "__main__":
    generate_report()
