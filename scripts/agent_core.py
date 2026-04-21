import asyncio
import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime

from scrape_threads import scrape_profile
from scrape_facebook import scrape_facebook_profile
from trading_agent import decide_action, get_action_weight

PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = "models/banini_model.pt"

# --- 模型延遲載入 (Lazy Load) 機制 ---
_tokenizer = None
_model = None
_device = None
_crawler_lock = None
last_seen_post_id = None

def get_crawler_lock():
    global _crawler_lock
    if _crawler_lock is None:
        _crawler_lock = asyncio.Lock()
    return _crawler_lock

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import BertTokenizer
        _tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        device = get_device()
        from transformers import BertForSequenceClassification
        _model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.to(device)
        _model.eval()
    return _model

def reload_model_weights():
    """ 觸發重新預載模型權重 """
    global _model
    if _model is not None:
        device = get_device()
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            _model.to(device)
            _model.eval()
            print("已重新載入模型權重")

def analyze_post_dimensions(text):
    sector = "未知"
    if any(k in text for k in ["台積", "半導", "聯發", "晶片", "GG", "2330"]):
        sector = "🔌 半導體/電子"
    elif any(k in text for k in ["長榮", "陽明", "航運", "海運", "萬海"]):
        sector = "🚢 航運"
    elif any(k in text for k in ["大盤", "台指", "熊", "牛", "ETF"]):
        sector = "📈 大盤/ETF/期貨"
        
    emotion = "平靜觀望 (無特殊訊號)"
    if any(k in text for k in ["停損", "賣出", "認賠", "空單", "put", "看衰"]):
        emotion = "🔪 認輸停損/看空 (底部已現👉反彈看漲)"
    elif any(k in text for k in ["救命", "慘", "被套", "不行了", "死抱", "持有"]):
        emotion = "😭 被套死抱中 (底部未到👉還有得跌)"
    elif any(k in text for k in ["買", "加碼", "看多", "上車", "噴", "賺", "舒服"]):
        emotion = "😎 看多買進/自信 (即將見頂👉高機率下跌)"
        
    return sector, emotion

def predict_contrarian(text):
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_device()
    
    inputs = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()
    return score

async def check_social_sentiment():
    lock = get_crawler_lock()
    if lock.locked():
        return "⏳ 系統正忙於處理前一個請求，請稍候重試。", None

    async with lock:
        try:
            users = ["banini31", "8zz_trade"]
            total_score = 0
            count = 0
            
            for u in users:
                results = await asyncio.wait_for(scrape_profile(u, max_scroll=1), timeout=60)
                own_posts = [p for p in results if p["author"] == u][:2]
                for p in own_posts:
                    total_score += predict_contrarian(p['text'])
                    count += 1
            
            if count == 0:
                return "❌ 無法獲取最新的群眾貼文。", None
                
            avg_score = total_score / count
            status = "🔥 極度貪婪 (危險)" if avg_score > 0.7 else ("❄️ 極度恐慌 (機會)" if avg_score < 0.3 else "⚖️ 中性盤整")
            
            report = "🌐 **社群大眾情緒溫度計**\n-----------------\n"
            report += f"分析樣本：{count} 則網紅貼文\n"
            report += f"情緒指數：`{avg_score:.1%}`\n"
            report += f"狀態判定：**{status}**\n"
            
            return report, avg_score
        except Exception as e:
            return f"❌ 量測市場情緒時失敗: {e}", None

async def generate_report(check_new_only=False):
    global last_seen_post_id
    lock = get_crawler_lock()
    
    if lock.locked():
        return "⏳ 系統正忙於處理前一個請求，請稍候 30 秒再試一次。", 0

    async with lock:
        try:
            username = "banini31"
            
            # 雙引擎同時啟動
            threads_task = scrape_profile(username, max_scroll=3)
            fb_task = scrape_facebook_profile(username, max_scroll=3)
            
            threads_res, fb_res = await asyncio.gather(
                asyncio.wait_for(threads_task, timeout=90),
                asyncio.wait_for(fb_task, timeout=90),
                return_exceptions=True
            )
            
            combined_posts = []
            
            if not isinstance(threads_res, Exception) and threads_res:
                for p in threads_res:
                    if p.get("author") == username:
                        p['source'] = "Threads"
                        combined_posts.append(p)
                
            if not isinstance(fb_res, Exception) and fb_res:
                for p in fb_res:
                    if p.get("author") == username:
                        p['source'] = "Facebook"
                        combined_posts.append(p)

            if not combined_posts: return None
            
            # 去重機制：依內文前 15 字元判斷
            unique_posts = []
            seen_snippets = set()
            
            for p in combined_posts:
                snippet = p['text'][:15].replace(" ", "").replace("\n", "")
                if snippet not in seen_snippets:
                    seen_snippets.add(snippet)
                    unique_posts.append(p)
                    
            if not unique_posts: return None
            
            latest_id = unique_posts[0]['id']
            if check_new_only and last_seen_post_id == latest_id:
                return None
                
            last_seen_post_id = latest_id
            
            report = f"📊 **巴逆逆 跨平台反指標分析戰報**\n"
            report += "--------------------------------\n"
            
            total_score = 0
            target_posts = unique_posts[:3]
            for i, post in enumerate(target_posts):
                text = post['text']
                source_tag = "🔵 FB" if post.get('source') == "Facebook" else "🟣 Threads"
                score = predict_contrarian(text)
                sector, emotion = analyze_post_dimensions(text)
                total_score += score
                
                report += f"{i+1}. [{source_tag}] 「{text[:40]}...」\n"
                
                action = decide_action(score, emotion)
                weight = get_action_weight(action)
                
                report += f"🎯 **分析標的:** {sector}\n"
                report += f"🗣️ **情境判定:** {emotion}\n"
                report += f"🧠 **反轉危險指數:** {score:.1%}\n"
                report += f"🤖 **Agent 動作:** `{action}` (權重: {weight})\n\n"

            avg_score = total_score / len(target_posts)
            if avg_score > 0.8:
                status = "🚨 **極度危險！強烈建議反向操作**"
            elif avg_score > 0.6:
                status = "⚠️ 微妙區間，建議謹慎觀察"
            else:
                status = "✅ 沒事，目前反轉機率低"
                
            report += "--------------------------------\n"
            report += f"📢 **綜合判定：**\n{status}\n"
            report += f"🧠 本地 AI 綜合反轉指數: {avg_score:.2%}\n"
            
            # --- 記錄入未卜先知庫 ---
            PENDING_FILE = "datasets/pending_validation.json"
            try:
                os.makedirs(os.path.dirname(PENDING_FILE), exist_ok=True)
                if os.path.exists(PENDING_FILE):
                    with open(PENDING_FILE, "r", encoding="utf-8") as f:
                        pending = json.load(f)
                else:
                    pending = []
                    
                existing_ids = {p['post_id'] for p in pending}
                for post in target_posts:
                    if post['id'] not in existing_ids:
                        text = post['text']
                        s, e = analyze_post_dimensions(text)
                        sc = predict_contrarian(text)
                        act = decide_action(sc, e)
                        
                        pending.append({
                            "post_id": post['id'],
                            "text": text,
                            "sector": s,
                            "emotion": e,
                            "timestamp": datetime.now().isoformat(),
                            "predicted_score": float(sc),
                            "action": act,
                            "action_weight": float(get_action_weight(act))
                        })
                with open(PENDING_FILE, "w", encoding="utf-8") as f:
                    json.dump(pending, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"寫入未卜先知庫失敗: {e}")
                
            return report, avg_score

        except asyncio.TimeoutError:
            return "🕒 爬蟲回應逾時 (Threads 可能正在阻擋或網路不穩)。", 0
        except Exception as e:
            return f"❌ 系統錯誤: {str(e)}", 0

if __name__ == "__main__":
    # 手動執行測試
    async def main():
        print("啟動獨立 AI 分析核心 (Lazy Load 測試)...")
        report_result = await generate_report()
        if report_result:
            rep, score = report_result
            safe_rep = rep.encode('cp950', 'replace').decode('cp950')
            print("\n" + safe_rep)
        else:
            print("目前沒抓到新資料。")
    asyncio.run(main())
