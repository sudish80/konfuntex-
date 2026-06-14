@echo off
cd /d "C:\Users\deuja\agent-brain\colab-agent"
python -c "
import sys, os
sys.path.insert(0, '.')
from colab.remote_executor import RemoteColabExecutor
from playwright.sync_api import sync_playwright

brave = r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'
bp = brave if os.path.isfile(brave) else None
r = RemoteColabExecutor(browser_path=bp)

print('=== Colab Login ===')
print('A browser window will open.')
print('1. Log into your Google account')
print('2. Navigate to https://colab.research.google.com')
print('3. Close the browser tab (not the whole window)')
print()
os.system('pause')

with sync_playwright() as pw:
    opts = {'user_data_dir': r.user_data_dir, 'headless': False, 'args': ['--no-sandbox']}
    if bp:
        opts['executable_path'] = bp
    browser = pw.chromium.launch_persistent_context(**opts)
    page = browser.new_page()
    page.goto('https://colab.research.google.com', timeout=60000)
    print('Browser open. Complete login then close the Colab tab.')
    os.system('pause')
    browser.close()

print(f'Session saved to: {r.user_data_dir}')
print('You can now use --executor remote with the agent.')
os.system('pause')
"
