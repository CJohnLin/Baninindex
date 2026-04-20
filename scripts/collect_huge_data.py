import asyncio
import json
import os
import sys
from datetime import datetime
# 匯入現有的 scrape_profile 函數
from scrape_threads import scrape_profile

# 目標帳號清單
TARGET_USERS = [
    "banini31",    
    "8zz_trade",   
    "8zz_8zz_8zz"
]

async def collect_huge_data():
    raw_output_dir = os.path.join("datasets", "raw")
    os.makedirs(raw_output_dir, exist_ok=True)
    
    total_posts = 0
    
    for username in TARGET_USERS:
        print(f"\n--- 正在開始收集 @{username} ---")
        try:
            # 增加捲動次數到 25 次以獲取更多歷史數據
            results = await scrape_profile(username, max_scroll=25)
            
            # 過濾出該帳號自己的貼文
            own_posts = [p for p in results if p["author"] == username]
            
            if own_posts:
                output_file = os.path.join(raw_output_dir, f"huge_{username}_{datetime.now().strftime('%Y%m%d')}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(own_posts, f, ensure_ascii=False, indent=2)
                
                print(f"✅ 成功！@{username}: 抓取到 {len(own_posts)} 篇貼文")
                print(f"📄 存檔路徑: {output_file}")
                total_posts += len(own_posts)
            else:
                print(f"⚠️ 警告：@{username} 沒有找到任何貼文。")
                
        except Exception as e:
            print(f"❌ 抓取 @{username} 時發生錯誤: {e}")

    print(f"\n========================================")
    print(f"任務完成！總計收集貼文數: {total_posts}")
    print(f"========================================")

if __name__ == "__main__":
    # 確保當前目錄在 scripts 下，或者能找到 scrape_threads
    # 建議在 banini 根目錄執行
    asyncio.run(collect_huge_data())
