@echo off
title Banini Telegram Bot
echo =========================================
echo 啟動巴逆逆反指標雷達 Telegram Bot
echo =========================================

:loop
echo [%time%] 正在啟動機器人...
python scripts\telegram_bot.py
echo.
echo [%time%] 機器人已停止或發生崩潰！
echo 將在 5 秒後自動重新啟動...
timeout /t 5
goto loop
