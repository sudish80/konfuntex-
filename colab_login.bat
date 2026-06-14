@echo off
cd /d "C:\Users\deuja\agent-brain\colab-agent"
python -c "
import sys, os
sys.path.insert(0, '.')
from colab.remote_executor import RemoteColabExecutor
from playwright.sync_api import sync_playwright

brave = r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'
brave_local = os.path.expandvars(r'%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe')
bp = brave if os.path.isfile(brave) else (brave_local if os.path.isfile(brave_local) else None)
r = RemoteColabExecutor(browser_path=bp)

print('=== Colab One-Click Login ===')
print(f'Browser: {\"Brave\" if bp else \"Chromium\"}')
print('1. A browser window will open')
print('2. Log into your Google account')
print('3. Navigate to colab.research.google.com')
print('4. Close the Colab tab (NOT the whole window)')
print()
os.system('pause')

with sync_playwright() as pw:
    opts = {'user_data_dir': r.user_data_dir, 'headless': False, 'args': ['--no-sandbox']}
    if bp:
        opts['executable_path'] = bp
    browser = pw.chromium.launch_persistent_context(**opts)
    page = browser.new_page()
    page.goto('https://colab.research.google.com', timeout=60000)
    print('Browser is open. Login, then close the Colab tab and press Enter here.')
    os.system('pause')
    browser.close()

print(f'\nSession saved to: {r.user_data_dir}')
print('You can now use: python cli.py run \"<goal>\" --executor remote')
os.system('pause')
"
