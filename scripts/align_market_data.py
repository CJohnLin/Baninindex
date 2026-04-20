import pandas as pd
import yfinance as yf
import os
import re
from datetime import datetime, timedelta

def extract_ticker(text):
    """
    從貼文中提取台股代號 (簡單範例：搜尋 4 位數字)。
    實務上可擴充比對常見權值股名稱。
    """
    match = re.search(r'(\d{4})', text)
    if match:
        return f"{match.group(1)}.TW"
    
    # 擴充常見名稱映射
    mapping = {
        "台積電": "2330.TW",
        "鴻海": "2317.TW",
        "聯發科": "2454.TW",
        "大盤": "^TWII",
        "小指": "YM=F", # 舉例
    }
    for name, ticker in mapping.items():
        if name in text:
            return ticker
    return None

def get_market_label(ticker, post_date_str, window_days=3):
    """
    計算貼文後 N 天的報酬率。
    """
    try:
        post_date = datetime.strptime(post_date_str, '%Y-%m-%d %H:%M:%S')
        start_date = post_date.strftime('%Y-%m-%d')
        end_date = (post_date + timedelta(days=window_days + 5)).strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty or len(data) < 2:
            return None, None
            
        # 計算未來 N 個交易日的變化 (以 Close 計)
        # 這裡取的是貼文後第一個交易日到第 N 個交易日
        price_at_post = data.iloc[0]['Close']
        price_future = data.iloc[min(window_days, len(data)-1)]['Close']
        
        price_at_post = float(price_at_post)
        price_future = float(price_future)
        
        pct_change = (price_future - price_at_post) / price_at_post
        return pct_change, data.iloc[0:window_days+1]['Close'].tolist()
    except Exception as e:
        print(f"抓取 {ticker} 資料失敗: {e}")
        return None, None

def align_data(input_csv, output_file):
    """將貼文與市場數據對齊"""
    df = pd.read_csv(input_csv)
    results = []

    print(f"正在對齊市場數據，共 {len(df)} 條貼文...")

    for _, row in df.iterrows():
        ticker = extract_ticker(row['text_clean'])
        if not ticker:
            continue
            
        change, price_series = get_market_label(ticker, row['timestamp'])
        
        if change is not None:
            results.append({
                'post_id': row['post_id'],
                'timestamp': row['timestamp'],
                'ticker': ticker,
                'text': row['text_clean'],
                'predicted_action': row['predicted_action'],
                'future_3d_return': change,
                'price_history': price_series,
                'is_contrarian_win': 1 if (row['predicted_action'] == 'BUY' and change < 0) or 
                                         (row['predicted_action'] == 'STOP_LOSS' and change > 0) else 0
            })

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"對齊完成！儲存至: {output_file}")
    print(f"成功對齊標的數: {len(final_df)}")

if __name__ == "__main__":
    PROCESSED_DIR = "datasets/processed"
    # 尋找最新的 processed csv
    csv_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.startswith('dataset_') and f.endswith('.csv')]
    
    if not csv_files:
        print("請先執行 python scripts/process_data.py 生成數據。")
    else:
        latest_csv = max(csv_files, key=os.path.getctime)
        output_path = os.path.join(PROCESSED_DIR, "aligned_training_data.csv")
        align_data(latest_csv, output_path)
