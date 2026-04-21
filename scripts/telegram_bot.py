import asyncio
import os
import json
import pandas as pd
from datetime import time, timezone, timedelta, datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()

# 匯入現有的模組
from auto_labeler import run_labeling
# 匯入解耦後的核心 AI 模組
import agent_core

# --- 配置區 ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("⚠️ 警告：找不到 TELEGRAM_TOKEN！請確認根目錄是否有 .env 檔案。")
PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = "models/banini_model.pt"
SUBSCRIBERS_FILE = "datasets/processed/subscribers.json"
DATASET_FILE = "datasets/processed/aligned_training_data.csv"

# 設定台北時區
TAIPEI_TZ = timezone(timedelta(hours=8))

# 將原本的資源鎖和模型交由 agent_core 接管

# --- 工具函數：訂閱管理 ---
def load_subscribers() -> dict:
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # 舊版 list 結構遷移至 dict
                    data = {str(cid): 0.60 for cid in data}
                return data
        except Exception:
            return {}
    return {}

def update_subscriber(chat_id, threshold=0.60):
    subs = load_subscribers()
    chat_id_str = str(chat_id)
    subs[chat_id_str] = threshold
    os.makedirs(os.path.dirname(SUBSCRIBERS_FILE), exist_ok=True)
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subs, f)

# 模型現在會在需要時由 agent_core 自動 Lazy Load

# --- Bot 擴充指令 ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏮 歡迎使用巴逆逆反指標雷達 2.0 🏮\n\n"
        "=== 實戰指令 ===\n"
        "/banini - 手動掃描最新貼文\n"
        "/manual <標的> - 模擬診斷特定股票\n"
        "/rank - 歷史標的冥燈勝率排行\n"
        "/sentiment - 掃描財經網紅群體大眾情緒\n\n"
        "=== 設定指令 ===\n"
        "/subscribe [門檻] - 訂閱推播 (預設0.6)\n"
        "/set_alert <門檻> - 設定觸發警報的分數門檻 (0~1)"
    )

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        threshold = float(context.args[0]) if context.args else 0.60
    except ValueError:
        await update.message.reply_text("❌ 格式錯誤。請輸入數字例如：/subscribe 0.75")
        return
        
    update_subscriber(update.message.chat_id, threshold)
    await update.message.reply_text(f"✅ 訂閱成功！您的專屬警報門檻設定為 {threshold:.0%}。\n當日台股收盤前後，只要分析指數高於此門檻，小幫手將主動通知您！")

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await subscribe(update, context)

async def print_rank(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(DATASET_FILE):
        await update.message.reply_text("📂 尚無足夠的歷史資料來計算排行榜。")
        return
        
    try:
        df = pd.read_csv(DATASET_FILE)
        if df.empty:
            await update.message.reply_text("📂 歷史資料庫目前為空。")
            return
            
        grouped = df.groupby('ticker').agg(
            total_mentions=('post_id', 'count'),
            contrarian_wins=('is_contrarian_win', 'sum')
        ).reset_index()
        
        grouped['win_rate'] = grouped['contrarian_wins'] / grouped['total_mentions']
        # 篩選至少被提及 2 次以上的標的，並依勝率排行
        ranked = grouped[grouped['total_mentions'] >= 2].sort_values(by='win_rate', ascending=False).head(5)
        
        if ranked.empty:
            ranked = grouped.sort_values(by='win_rate', ascending=False).head(5)
            
        msg = "🏆 **巴逆逆歷史冥燈排行榜 (Top 5)**\n\n"
        for i, row in ranked.iterrows():
            msg += f"🏅 **{row['ticker']}**\n"
            msg += f"   提及次數: {row['total_mentions']} 次\n"
            msg += f"   反轉勝率: `{row['win_rate']:.1%}`\n\n"
            
        await update.message.reply_text(msg, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ 計算排行榜時發生錯誤: {e}")

async def manual_diagnose(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ 指令錯誤。請提供標的，例如： `/manual 2330` 或 `/manual 航運`")
        return
        
    ticker = " ".join(context.args)
    text_bull = f"重倉買進 {ticker}，準備要噴了，大家快上車！"
    text_bear = f"受不了了，{ticker} 這個垃圾，我認賠停損出場！"
    
    score_bull = agent_core.predict_contrarian(text_bull)
    score_bear = agent_core.predict_contrarian(text_bear)
    
    msg = f"🧪 **虛擬情境模擬：{ticker}**\n"
    msg += f"如果您看見她發布看多貼文：\n"
    msg += f"「{text_bull}」\n"
    msg += f"👉 觸發反轉向下機率：`{score_bull:.1%}`\n\n"
    
    msg += f"如果您看見她發布停損貼文：\n"
    msg += f"「{text_bear}」\n"
    msg += f"👉 觸發觸底反彈機率：`{score_bear:.1%}`\n"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def check_sentiment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📡 正在掃描財經網紅群體情緒 (這需要大約一分鐘)...")
    
    report, _ = await agent_core.check_social_sentiment()
    if not report:
        await msg.edit_text("❌ 無法獲取最新資料。")
    else:
        await msg.edit_text(report, parse_mode='Markdown')

# --- 分析報告 ---
async def force_check_banini(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 手動觸發：正在深度爬取與 AI 分析中...")
    result = await agent_core.generate_report()
    if not result:
        await msg.edit_text("目前沒抓到新資料。")
        return
        
    if isinstance(result, tuple):
        report, _ = result
    else:
        report = result
        
    await msg.edit_text(report, parse_mode='Markdown')

# --- Agent 虛擬錢包功能 ---
def load_wallet():
    WALLET_FILE = "datasets/processed/wallet.json"
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE, "r") as f:
            return json.load(f)
    return {"balance": 1000000.0, "total_reward_pct": 0.0, "trades": 0}

async def handle_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wallet = load_wallet()
    balance = wallet["balance"]
    trades = wallet["trades"]
    reward = wallet["total_reward_pct"] * 100
    
    text = (
        "💼 **Agent 模擬交易虛擬戶頭 (單位: NTD)**\n\n"
        f"🏦 目前總資產: `{balance:,.0f}` 元\n"
        f"📊 執行決策數: `{trades}` 次\n"
        f"📈 Agent 累積獲利率 (RL Reward): `{reward:+.2f}%`\n\n"
        "_此帳戶根據每次模型給出的做多/做空/觀望訊號，並與 3天後的真實市場驗證對齊後所虛擬結算之結果。_"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

# --- 排程工作 ---
async def midnight_labeler_job(context: ContextTypes.DEFAULT_TYPE):
    """每日凌晨執行事後諸葛腳本，驗證 3 天前的預測"""
    loop = asyncio.get_running_loop()
    new_data_count = await loop.run_in_executor(None, run_labeling)
    if new_data_count > 0:
        print(f"已正式學習 {new_data_count} 筆新資料！")

async def weekly_retrain_job(context: ContextTypes.DEFAULT_TYPE):
    """每週背景重訓模型"""
    print("啟動背景重訓引擎...")
    subs = load_subscribers()
    for chat_id_str in subs:
        try:
            await context.bot.send_message(chat_id=chat_id_str, text="⚙️ **全自動進化啟動**\nAI 正在吸收您這週的最新市場數據，即將在背景展開神經網路重建，預計數分鐘後完成...")
        except:
            pass

    # 非同步執行訓練腳本
    process = await asyncio.create_subprocess_exec(
        'python', 'scripts/train_model.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        # 重訓成功，進行熱重載
        agent_core.reload_model_weights()
        for chat_id_str in subs:
            try:
                await context.bot.send_message(chat_id=chat_id_str, text="🎉 **自我進化完成**\n模型已成功載入最新權重 (Hot Reload)，您現在擁有一個更聰明的巴逆逆雷達了！")
            except:
                pass
    else:
        print(f"背景重訓失敗: {stderr.decode()}")
        for chat_id_str in subs:
            try:
                await context.bot.send_message(chat_id=chat_id_str, text="⚠️ **進化失敗**\n背景重新訓練發生錯誤，目前仍維持前朝版本繼續運作。")
            except:
                pass

async def auto_push_job(context: ContextTypes.DEFAULT_TYPE):
    print("啟動自動推播檢查機制...")
    result = await agent_core.generate_report(check_new_only=False)
    if not result: return
    
    if isinstance(result, tuple):
        report_text, avg_score = result
    else:
        report_text, avg_score = result, 0.0 # Error messages
        
    subs = load_subscribers()
    for chat_id_str, threshold in subs.items():
        if avg_score >= threshold or "過時" in report_text or "錯誤" in report_text:
            try:
                await context.bot.send_message(chat_id=chat_id_str, text="⏰ **[收盤前後戰況掃描]**\n\n" + report_text, parse_mode='Markdown')
            except Exception as e:
                print(f"無法發送給 {chat_id_str}: {e}")

if __name__ == "__main__":
    print("初始化 Telegram 機器人核心...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # 註冊指令
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("banini", force_check_banini))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("set_alert", set_alert))
    app.add_handler(CommandHandler("rank", print_rank))
    app.add_handler(CommandHandler("manual", manual_diagnose))
    app.add_handler(CommandHandler("sentiment", check_sentiment))
    app.add_handler(CommandHandler("wallet", handle_wallet))
    
    # 註冊排程 (使用台北時間)
    jq = app.job_queue
    jq.run_daily(auto_push_job, time(hour=13, minute=0, tzinfo=TAIPEI_TZ))
    jq.run_daily(auto_push_job, time(hour=13, minute=45, tzinfo=TAIPEI_TZ))
    
    # 持續學習：每日凌晨 2 點驗證資料，每週日凌晨 3 點背景重訓
    jq.run_daily(midnight_labeler_job, time(hour=2, minute=0, tzinfo=TAIPEI_TZ))
    jq.run_daily(weekly_retrain_job, time(hour=3, minute=0, tzinfo=TAIPEI_TZ), days=(6,))  # 6 = 星期日

    print("🚀 Telegram Bot (終極版：排行榜/情緒/持續進化) 已上線！ 等候操作中...")
    app.run_polling()
