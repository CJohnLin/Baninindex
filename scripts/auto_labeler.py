import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

PENDING_FILE = "datasets/pending_validation.json"
TRAINING_FILE = "datasets/processed/aligned_training_data.csv"
WALLET_FILE = "datasets/processed/wallet.json"

def get_ticker_for_sector(sector):
    if "半導體" in sector: return "2330.TW"
    if "航運" in sector: return "2603.TW"
    return "^TWII"  # 預設大盤

def run_labeling():
    print("啟動自動延遲打標系統 (Auto Labeler)...")
    if not os.path.exists(PENDING_FILE):
        print("沒有待驗證的貼文。")
        return 0
        
    with open(PENDING_FILE, "r") as f:
        try:
            pending_posts = json.load(f)
        except:
            pending_posts = []
            
    if not pending_posts:
        return 0

    now = datetime.now()
    validated_posts = []
    remaining_posts = []
    
    for post in pending_posts:
        post_time = datetime.fromisoformat(post['timestamp'])
        days_passed = (now - post_time).days
        
        # 滿 3 天才能驗證
        if days_passed >= 3:
            ticker = get_ticker_for_sector(post['sector'])
            
            try:
                # 抓取這幾天的數據
                start_date = post_time.strftime("%Y-%m-%d")
                end_date_time = post_time + timedelta(days=5) # 抓稍多幾天避免假日
                end_date = end_date_time.strftime("%Y-%m-%d")
                
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(df) >= 2:
                    start_price = df['Close'].iloc[0]
                    # 選擇大約3天後的價格 (假設有至少2天交易日)
                    end_price = df['Close'].iloc[-1]
                    
                    # Ensure scalar value (yf.download sometimes returns pd.Series or pd.DataFrame)
                    if isinstance(start_price, pd.Series): start_price = start_price.iloc[0]
                    if isinstance(end_price, pd.Series): end_price = end_price.iloc[0]
                    
                    return_rate = (end_price - start_price) / start_price
                    
                    is_contrarian_win = 0
                    emotion = post.get('emotion', '')
                    
                    # 包含「看跌/還有得跌」字眼的情緒（代表反向預期市場會跌）：被套、看多
                    if ("跌" in emotion) and return_rate < 0:
                        is_contrarian_win = 1
                    # 包含「看漲/反彈」字眼的情緒（代表反向預期市場會漲）：停損、看空
                    elif ("漲" in emotion) and return_rate > 0:
                        is_contrarian_win = 1
                        
                    # --- 計算 RL Agent Reward ---
                    action_weight = post.get('action_weight', 0.0)
                    reward_pct = float(return_rate * action_weight)
                    action_str = post.get('action', '未知')
                    # -----------------------------
                        
                    # 已經成功驗證並打標
                    validated_posts.append({
                        "post_id": post['post_id'],
                        "text": post['text'],
                        "timestamp": post['timestamp'],
                        "ticker": ticker,
                        "return_rate": return_rate,
                        "is_contrarian_win": is_contrarian_win,
                        "action": action_str,
                        "reward_pct": reward_pct
                    })
                    print(f"✅ 驗證成功貼文 {post['post_id']} -> Agent動作: {action_str}, 獲得Reward: {reward_pct:+.2%}")
                else:
                    # 抓不到數據暫時留著
                    remaining_posts.append(post)
            except Exception as e:
                print(f"⚠️ 驗證貼文 {post['post_id']} 時出錯: {e}")
                remaining_posts.append(post)
        else:
            remaining_posts.append(post)
            
    # 將已驗證的寫入訓練集
    if validated_posts:
        new_df = pd.DataFrame(validated_posts)
        if os.path.exists(TRAINING_FILE):
            old_df = pd.read_csv(TRAINING_FILE)
            merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['post_id'])
            merged_df.to_csv(TRAINING_FILE, index=False)
        else:
            os.makedirs(os.path.dirname(TRAINING_FILE), exist_ok=True)
            new_df.to_csv(TRAINING_FILE, index=False)
            
        # 更新虛擬錢包 (RL Settlement)
        if os.path.exists(WALLET_FILE):
            with open(WALLET_FILE, "r") as f:
                wallet = json.load(f)
        else:
            wallet = {"balance": 1000000.0, "total_reward_pct": 0.0, "trades": 0}
            
        for rp in validated_posts:
            r = rp['reward_pct']
            if r != 0.0:  # 觀望不動作不算 trade
                wallet['balance'] *= (1 + r)
                wallet['total_reward_pct'] += r
                wallet['trades'] += 1
                
        with open(WALLET_FILE, "w") as f:
            json.dump(wallet, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 成功將 {len(validated_posts)} 筆新黃金樣本併入訓練庫，並結算至虛擬錢包。")

    # 更新待驗證清單
    with open(PENDING_FILE, "w") as f:
        json.dump(remaining_posts, f, ensure_ascii=False, indent=2)
        
    return len(validated_posts)

if __name__ == "__main__":
    run_labeling()
