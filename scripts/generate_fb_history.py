import pandas as pd
import yfinance as yf
import random
import os
import uuid
import json
from datetime import datetime, timedelta
import sys

# 將腳本執行目錄切到根目錄以正確匯入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import agent_core
    from trading_agent import decide_action, get_action_weight
except ImportError:
    print("無法載入 agent_core，請確保在專案根目錄下執行。")
    sys.exit(1)

OUTPUT_FILE = "datasets/processed/aligned_training_data.csv"

TICKERS = {
    "2330.TW": ["台積電", "GG", "2330", "神山"],
    "2317.TW": ["鴻海", "海公公", "2317"],
    "2454.TW": ["聯發科", "發哥", "2454"],
    "^TWII": ["大盤", "台股", "指數"],
    "2603.TW": ["長榮", "航海王", "2603"],
    "2609.TW": ["陽明", "2609"],
    "2303.TW": ["聯電", "二哥", "2303"]
}

TEMPLATES = {
    "BUY": [
        "重壓 {name}，準備起飛啦！",
        "今天 {name} 有夠甜，加碼兩張買進！",
        "大盤不看好，但我看好 {name}",
        "看到 {name} 這個價位，不歐印對不起自己",
        "{name} 買了買了，舒服！"
    ],
    "SHORT": [
        "{name} 爛死了，直接空爆它",
        "看到 {name} 就討厭，今天券空爽賺",
        "這波跌勢確定了，放空 {name} 等數錢",
    ],
    "STOP_LOSS": [
        "受不了 {name} 了一路跌，我停損認賠...",
        "砍掉 {name}，太痛了，認輸",
        "玩 {name} 玩到心累，全部賣出，刪APP",
        "{name} 殺成這樣我受不了了，出清停損啦"
    ],
    "TRAPPED": [
        "被 {name} 套牢QQ，只能當存股了",
        "滿手 {name} 水下，還有救嗎",
        "死抱 {name} 不賣，只要不賣就不算賠...",
        "看著 {name} 一路崩，我也只能抱著裝死"
    ]
}

def generate_fb_posts():
    start_date = datetime(2024, 4, 1)
    end_date = datetime.now() - timedelta(days=4) # 確保有未來 3 天股市資料
    
    current_date = start_date
    posts = []
    
    print("正在生成 2024-04 至今的歷史發文與對應歷史股市報酬...")
    
    while current_date <= end_date:
        # 決定今天發不發文 (頻率約 3-4 天一篇)
        if random.random() < 0.3:
            ticker = random.choice(list(TICKERS.keys()))
            name = random.choice(TICKERS[ticker])
            
            emotion_type = random.choice(list(TEMPLATES.keys()))
            text = random.choice(TEMPLATES[emotion_type]).format(name=name)
            
            post_id = f"fb_mock_{uuid.uuid4().hex[:8]}"
            date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            posts.append({
                "post_id": post_id,
                "text": text,
                "timestamp": date_str,
                "ticker": ticker,
                "emotion_type": emotion_type
            })
            
        current_date += timedelta(days=1)
        
    print(f"生成了 {len(posts)} 篇 FB 模擬發文。準備對齊 YFinance 真實 K 線...")
    return posts

def enrich_and_align_posts(posts):
    validated_posts = []
    
    for i, post in enumerate(posts):
        ticker = post['ticker']
        post_time = datetime.strptime(post['timestamp'], "%Y-%m-%d %H:%M:%S")
        emotion_type = post['emotion_type']
        
        # 抓 yfinance (發文日往後抓 7 天，取最靠近的 3 交易日後)
        start_date = post_time.strftime("%Y-%m-%d")
        end_date = (post_time + timedelta(days=10)).strftime("%Y-%m-%d")
        
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) >= 2:
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[min(3, len(df)-1)]
                
                if isinstance(start_price, pd.Series): start_price = start_price.iloc[0]
                if isinstance(end_price, pd.Series): end_price = end_price.iloc[0]
                
                start_price = float(start_price)
                end_price = float(end_price)
                
                return_rate = (end_price - start_price) / start_price
                
                # 反指標判定邏輯
                is_contrarian_win = 0
                if emotion_type in ["BUY", "TRAPPED"] and return_rate < 0:
                    is_contrarian_win = 1
                elif emotion_type in ["SHORT", "STOP_LOSS"] and return_rate > 0:
                    is_contrarian_win = 1
                    
                # 取得 Agent 預測分數
                text = post['text']
                score = agent_core.predict_contrarian(text)
                sector, emotion_str = agent_core.analyze_post_dimensions(text)
                action = decide_action(score, emotion_str)
                action_weight = get_action_weight(action)
                reward_pct = return_rate * action_weight
                
                validated_posts.append({
                    "post_id": post['post_id'],
                    "text": text,
                    "timestamp": post['timestamp'],
                    "ticker": ticker,
                    "return_rate": return_rate,
                    "is_contrarian_win": is_contrarian_win,
                    "action": action,
                    "reward_pct": reward_pct
                })
        except Exception as e:
            continue
            
        if (i+1) % 10 == 0:
            print(f"進度: {i+1}/{len(posts)} ...")
            
    return validated_posts

if __name__ == "__main__":
    posts = generate_fb_posts()
    aligned = enrich_and_align_posts(posts)
    
    new_df = pd.DataFrame(aligned)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)
        merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['post_id'])
        merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    else:
        new_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
    print(f"\n成功寫入 {len(new_df)} 筆「真實股價對齊」歷史數據！")
    print(f"現在您可以執行報表生成，或執行 python scripts/train_model.py 進行訓練。")
