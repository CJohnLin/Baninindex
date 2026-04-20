import asyncio
import os
import json
import torch
from datetime import time, timezone, timedelta
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# 匯入現有的爬蟲模組
from scrape_threads import scrape_profile

# --- 配置區 ---
TELEGRAM_TOKEN = "8792257959:AAHiM2OJhvBqjE4b4AnsBjDboxgaUDiMHQY"
PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = "models/banini_model.pt"
SUBSCRIBERS_FILE = "datasets/processed/subscribers.json"

# 設定台北時區
TAIPEI_TZ = timezone(timedelta(hours=8))

# 狀態記錄
last_seen_post_id = None

# --- 工具函數：訂閱管理 ---
def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            return json.load(f)
    return []

def add_subscriber(chat_id):
    subs = load_subscribers()
    if chat_id not in subs:
        subs.append(chat_id)
        os.makedirs(os.path.dirname(SUBSCRIBERS_FILE), exist_ok=True)
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(subs, f)
        return True
    return False

# --- 模型載入 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ 成功載入本地模型: {MODEL_PATH}")
else:
    print(f"⚠️ 找不到模型權重，將使用初始模型進行預測。")

model.to(device)
model.eval()

def analyze_post_dimensions(text):
    """提取多維度分析 (情緒、產業)"""
    sector = "未知"
    if any(k in text for k in ["台積", "半導", "聯發", "晶片", "GG"]):
        sector = "🔌 半導體/電子"
    elif any(k in text for k in ["長榮", "陽明", "航運", "海運", "萬海"]):
        sector = "🚢 航運"
    elif any(k in text for k in ["大盤", "台指", "熊", "牛"]):
        sector = "📈 大盤/期貨"
        
    emotion = "平靜"
    if any(k in text for k in ["救命", "慘", "被套", "不行了", "公園"]):
        emotion = "😭 極度崩潰 (強反轉反指標)"
    elif any(k in text for k in ["賺", "噴", "爽", "舒服", "數錢"]):
        emotion = "😎 自信得意 (即將見頂反轉)"
        
    return sector, emotion

def predict_contrarian(text):
    """預測單則貼文的反指標勝率"""
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

# --- Bot 指令回應 ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏮 歡迎使用巴逆逆反指標雷達 (本地 AI 版) 🏮\n\n"
        "指令列表：\n"
        "/banini - 手動掃描最新情境\n"
        "/subscribe - 訂閱自動推播 (每日台股收盤前後)"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if add_subscriber(chat_id):
        await update.message.reply_text("✅ 訂閱成功！小幫手將會在台股收盤前 (13:00) 與 收盤後 (13:45) 自動幫您推播最新神諭。")
    else:
        await update.message.reply_text("您之前已經訂閱過了喔！")

async def force_check_banini(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 手動觸發：正在爬取與 AI 分析中...")
    report = await generate_report()
    if report:
        await msg.edit_text(report, parse_mode='Markdown')
    else:
        await msg.edit_text("目前沒抓到新資料。")

# --- 自動化報告生成邏輯 ---
async def generate_report(check_new_only=False):
    global last_seen_post_id
    try:
        username = "banini31"
        results = await scrape_profile(username, max_scroll=3)
        own_posts = [p for p in results if p["author"] == username]
        
        if not own_posts: return None

        latest_id = own_posts[0]['id']
        if check_new_only and last_seen_post_id == latest_id:
            # 沒有新貼文，保持沉默
            return None
            
        last_seen_post_id = latest_id

        report = f"📊 **巴逆逆 (8zz) 反指標分析戰報**\n"
        report += "--------------------------------\n"
        
        total_score = 0
        target_posts = own_posts[:3]
        for i, post in enumerate(target_posts):
            text = post['text']
            score = predict_contrarian(text)
            sector, emotion = analyze_post_dimensions(text)
            total_score += score
            
            stars = "🔥" * int(score * 10) if score > 0.5 else "❄️" * int((1-score)*5)
            
            report += f"{i+1}. 「{text[:40]}...」\n"
            report += f"   🏷️ 產業: {sector}\n"
            report += f"   🎭 狀態: {emotion}\n"
            report += f"   🎯 冥燈信心: {score:.2%} {stars}\n\n"

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
        report += "\n*以上分析由本地顯卡驅動，不構成投資建議*"
        return report

    except Exception as e:
        print(f"分析失敗: {e}")
        return None

# --- 排程工作 ---
async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    """為所有訂閱者推播報告"""
    print("啟動自動推播檢查機制...")
    report = await generate_report(check_new_only=False) # 即使沒更新也推播收盤戰況
    if not report: 
        print("查無報告。")
        return
        
    subs = load_subscribers()
    for chat_id in subs:
        try:
            await context.bot.send_message(chat_id=chat_id, text="⏰ **[收盤前後戰況掃描]**\n\n" + report, parse_mode='Markdown')
        except Exception as e:
            print(f"無法發送給 {chat_id}: {e}")

if __name__ == "__main__":
    print("初始化 Telegram 機器人核心...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # 註冊指令
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("banini", force_check_banini))
    
    # 註冊排程 (使用台北時間)
    jq = app.job_queue
    # 收盤前 13:00 推播
    jq.run_daily(auto_push_job, time=time(hour=13, minute=0, tzinfo=TAIPEI_TZ))
    # 收盤後 13:45 再次結算
    jq.run_daily(auto_push_job, time=time(hour=13, minute=45, tzinfo=TAIPEI_TZ))

    print("🚀 Telegram Bot (訂閱與排程強化版) 已上線！ 等候操作中...")
    app.run_polling()
