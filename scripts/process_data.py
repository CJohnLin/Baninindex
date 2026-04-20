import json
import os
import re
import pandas as pd
from datetime import datetime, timezone, timedelta

def clean_text(text):
    """移除網址、HTML 標籤及多餘空格"""
    text = re.sub(r'http\S+', '', text)  # 移除 URL
    text = re.sub(r'<.*?>', '', text)    # 移除 HTML 標籤
    text = re.sub(r'\s+', ' ', text).strip() # 正規化空格
    return text

def parse_action(text):
    """
    初步辨識動作標籤 (可進階串接 LLM 作更精準標註)。
    基於 banini 的核心邏輯：
    """
    text = text.lower()
    if any(k in text for k in ['買', '加碼', '進場', '做多']):
        return 'BUY'
    if any(k in text for k in ['停損', '認賠', '砍掉', '出清']):
        return 'STOP_LOSS'
    if any(k in text for k in ['被套', '慘', '住公園', '救命']):
        return 'TRAPPED'
    if any(k in text for k in ['空', 'put', '跌']):
        return 'SHORT'
    return 'UNKNOWN'

def process_threads_json(input_file, output_dir):
    """讀取爬蟲 JSON 並轉換為訓練格式"""
    if not os.path.exists(input_file):
        print(f"Error: 找不到檔案 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_list = []
    
    for post in data:
        raw_text = post.get('text', '')
        cleaned = clean_text(raw_text)
        if not cleaned: continue

        # 轉換時間 (Unix timestamp to UTC+8)
        dt = datetime.fromtimestamp(post.get('taken_at', 0), tz=timezone.utc)
        tw_time = dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')

        processed_list.append({
            'post_id': post.get('id'),
            'timestamp': tw_time,
            'text_raw': raw_text,
            'text_clean': cleaned,
            'predicted_action': parse_action(cleaned),
            'likes': post.get('likes', 0),
            'reply_count': post.get('reply_count', 0)
        })

    # 1. 儲存為 CSV (方便人類閱讀與手動標註資料)
    df = pd.DataFrame(processed_list)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f'dataset_{timestamp_str}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 2. 儲存為 JSONL (機器學習常用格式)
    jsonl_path = os.path.join(output_dir, f'dataset_{timestamp_str}.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in processed_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"處理完成！")
    print(f"CSV 存檔路徑: {csv_path}")
    print(f"JSONL 存檔路徑: {jsonl_path}")
    print(f"總計處理貼文數: {len(processed_list)}")

if __name__ == "__main__":
    # 預設處理 datasets/raw 中的最新檔案，或是指定檔案
    RAW_DIR = "datasets/raw"
    PROCESSED_DIR = "datasets/processed"
    
    # 這裡可以手動指定檔案，或自動抓取 raw 目錄下最近的一個檔案
    try:
        files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        if not files:
            print("目前 datasets/raw 中沒有 .json 檔案。請先執行 /banini 抓取資料。")
        else:
            latest_file = max(files, key=os.path.getctime)
            process_threads_json(latest_file, PROCESSED_DIR)
    except Exception as e:
        print(f"執行時發生錯誤: {e}")
