import json
import asyncio
import sys
import uuid
from typing import Dict, List
from playwright.async_api import async_playwright
from parsel import Selector

async def scrape_facebook_profile(username: str, max_scroll: int = 2) -> List[Dict]:
    """Scrape a Facebook profile page for recent posts."""
    posts = []
    seen_texts = set()

    async with async_playwright() as pw:
        # 模擬行動特徵，繞過部分嚴格的 Desktop Login Wall
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="zh-TW",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        )
        page = await context.new_page()

        # PWA 版通常比 www 容易抓取純文字
        url = f"https://m.facebook.com/{username}"
        print(f"[fb-scraper] navigating to {url}", file=sys.stderr)
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"[fb-scraper] warning: page load timed out or failed: {e}", file=sys.stderr)

        for i in range(max_scroll):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(2000)
            print(f"[fb-scraper] scroll {i + 1}/{max_scroll}", file=sys.stderr)

        html = await page.content()
        selector = Selector(text=html)
        
        # 捕捉 FB 手機版的常見貼文區塊 (data-ft 或 story_body_container)
        nodes = selector.xpath("//div[@data-ft]")
        if nodes:
            for node in nodes:
                text_content = " ".join(node.xpath(".//text()").getall()).strip()
                if len(text_content) > 15 and text_content not in seen_texts:
                    seen_texts.add(text_content)
                    posts.append({
                        "id": f"fb_node_{uuid.uuid4().hex[:8]}",
                        "text": text_content,
                        "author": username,
                        "source": "Facebook"
                    })
        else:
            # Fallback 策略：尋找自動排版文字區 block
            for text_node in selector.xpath("//div[@dir='auto']").getall():
                sub_sel = Selector(text=text_node)
                clean_text = " ".join(sub_sel.xpath("//text()").getall()).strip()
                if clean_text and len(clean_text) > 20 and clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    posts.append({
                        "id": f"fb_auto_{uuid.uuid4().hex[:8]}",
                        "text": clean_text,
                        "author": username,
                        "source": "Facebook"
                    })

        await browser.close()

    # 過濾常見介面雜訊
    blocked_keywords = ["留言", "分享", "讚", "更多", "時間", "登入", "忘記密碼", "加入"]
    filtered_posts = []
    for p in posts:
        txt = p["text"]
        # 過短且含有雜訊關鍵字的，極有可能是按鈕
        if len(txt) > 20 and not any(len(txt) < 30 and bk in txt for bk in blocked_keywords):
            filtered_posts.append(p)
            
    return filtered_posts

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scrape_facebook.py <username> [max_scroll]", file=sys.stderr)
        sys.exit(1)

    username = sys.argv[1]
    max_scroll = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    results = asyncio.run(scrape_facebook_profile(username, max_scroll))

    if results:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[fb-scraper] total posts extracted: {len(results)}", file=sys.stderr)
    else:
        print("[]")
        print("[fb-scraper] no posts found or blocked by Login Wall", file=sys.stderr)

if __name__ == "__main__":
    main()
