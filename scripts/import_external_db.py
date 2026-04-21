import sqlite3
import pandas as pd
import os
import sys

# 將腳本執行目錄切到根目錄以正確匯入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import agent_core
    from trading_agent import decide_action, get_action_weight
except ImportError:
    print("無法載入 agent_core，請確保在專案根目錄下執行。")
    sys.exit(1)

DB_PATH = "datasets/raw/banini-public.db"
OUTPUT_FILE = "datasets/processed/aligned_training_data.csv"

def import_external_database():
    if not os.path.exists(DB_PATH):
        print(f"錯誤: 找不到資料庫 {DB_PATH}")
        return
        
    print(f"正在讀取並轉譯 SQLite 資料庫: {DB_PATH} ...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 提取 prediction 與 snapshot (取最接近 3 天的 snapshot 作為 future return)
    # 利用 subquery 抓出未來三日報酬率
    query = """
    SELECT 
        p.post_id, 
        p.symbol_code, 
        p.reasoning, 
        p.created_at, 
        (SELECT change_pct_close FROM price_snapshots WHERE prediction_id = p.id AND day_number >= 3 ORDER BY day_number ASC LIMIT 1) as future_3d_return
    FROM predictions p
    WHERE future_3d_return IS NOT NULL AND p.reasoning IS NOT NULL AND p.reasoning != ''
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("資料庫中無可用資料。")
        return
        
    print(f"從外部資料庫提取出 {len(df)} 筆具備真實歷史行情的文字樣本！準備對齊為 DRL 格式...")
    
    validated_posts = []
    
    for i, row in df.iterrows():
        # 對齊基礎資料
        text = str(row['reasoning'])
        ticker = f"{row['symbol_code']}.TW" if not str(row['symbol_code']).endswith('.TW') else row['symbol_code']
        post_id = f"ext_{row['post_id']}"  # 加上 ext 前綴避免和原本碰撞
        return_rate = float(row['future_3d_return']) / 100.0  # SQLite 中可能是 % (例如 1.5 代表 1.5%), 轉換為 0.015
        
        # 呼叫 Agent 核心邏輯
        score = agent_core.predict_contrarian(text)
        sector, emotion_str = agent_core.analyze_post_dimensions(text)
        action = decide_action(score, emotion_str)
        action_weight = get_action_weight(action)
        
        # 計算 冥燈勝率標籤
        is_contrarian_win = 0
        if ("跌" in emotion_str or "空" in emotion_str) and return_rate < 0:
            is_contrarian_win = 1
        elif ("漲" in emotion_str or "多" in emotion_str) and return_rate > 0:
            is_contrarian_win = 1
            
        # RL Reward
        reward_pct = return_rate * action_weight
        
        validated_posts.append({
            "post_id": post_id,
            "text": text,
            "timestamp": row['created_at'],
            "ticker": ticker,
            "return_rate": return_rate,
            "is_contrarian_win": is_contrarian_win,
            "action": action,
            "reward_pct": reward_pct
        })
        
        if (i+1) % 50 == 0:
            print(f"進度: {i+1}/{len(df)} 筆已轉換...")
            
    # 合併進訓練集
    new_df = pd.DataFrame(validated_posts)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)
        merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['post_id'])
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        total_len = len(merged_df)
    else:
        new_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        total_len = len(new_df)
        
    print(f"\n外來物種成功融入！寫入了 {len(new_df)} 筆外部社群黃金樣本。")
    print(f"目前訓練集 `aligned_training_data.csv` 總筆數高達：{total_len} 筆！")
    print("立刻執行 `python scripts/train_model.py` 讓大腦吸收這些功力吧！")

if __name__ == '__main__':
    import_external_database()
