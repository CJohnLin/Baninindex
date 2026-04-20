import pandas as pd
import os

def create_synthetic_data():
    os.makedirs("datasets/processed", exist_ok=True)
    
    data = [
        {"post_id": "1", "timestamp": "2026-04-10 10:00:00", "ticker": "2330.TW", "text": "台積電今天看來不錯，加碼買進！", "predicted_action": "BUY", "future_3d_return": -0.05, "is_contrarian_win": 1},
        {"post_id": "2", "timestamp": "2026-04-11 11:00:00", "ticker": "2317.TW", "text": "鴻海跌成這樣，停損認賠了...", "predicted_action": "STOP_LOSS", "future_3d_return": 0.08, "is_contrarian_win": 1},
        {"post_id": "3", "timestamp": "2026-04-12 12:00:00", "ticker": "^TWII", "text": "大盤要起飛了，快點進場做多", "predicted_action": "BUY", "future_3d_return": 0.02, "is_contrarian_win": 0},
        {"post_id": "4", "timestamp": "2026-04-13 13:00:00", "ticker": "2454.TW", "text": "聯發科被套牢，只能含淚續抱QQ", "predicted_action": "TRAPPED", "future_3d_return": -0.03, "is_contrarian_win": 1},
        {"post_id": "5", "timestamp": "2026-04-14 14:00:00", "ticker": "2330.TW", "text": "台積電太弱了，準備放空", "predicted_action": "SHORT", "future_3d_return": 0.06, "is_contrarian_win": 1},
        {"post_id": "6", "timestamp": "2026-04-15 15:00:00", "ticker": "2317.TW", "text": "加碼兩張！不信不會漲！", "predicted_action": "BUY", "future_3d_return": -0.01, "is_contrarian_win": 1},
        {"post_id": "7", "timestamp": "2026-04-16 16:00:00", "ticker": "^TWII", "text": "真的受不了，把持股全部出清", "predicted_action": "STOP_LOSS", "future_3d_return": 0.01, "is_contrarian_win": 1},
        {"post_id": "8", "timestamp": "2026-04-17 17:00:00", "ticker": "2454.TW", "text": "大家趕快買，穩賺不賠", "predicted_action": "BUY", "future_3d_return": -0.07, "is_contrarian_win": 1},
        {"post_id": "9", "timestamp": "2026-04-18 18:00:00", "ticker": "2330.TW", "text": "今天又被外資割韭菜", "predicted_action": "UNKNOWN", "future_3d_return": 0.00, "is_contrarian_win": 0},
        {"post_id": "10", "timestamp": "2026-04-19 19:00:00", "ticker": "2317.TW", "text": "做多鴻海，月底等著數錢", "predicted_action": "BUY", "future_3d_return": 0.05, "is_contrarian_win": 0},
    ] * 5  # 複製 5 遍讓樣本數達到 50 筆
    
    df = pd.DataFrame(data)
    output_path = "datasets/processed/aligned_training_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已建立合成測試數據！共 {len(df)} 筆，儲存至 {output_path}")

if __name__ == "__main__":
    create_synthetic_data()
