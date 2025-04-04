from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir='chatgpt_session',
        headless=False
    )
    page = browser.new_page()
    page.goto("https://chat.openai.com/")
    input("Log in manually, then press Enter here to save the session...")
    browser.close()