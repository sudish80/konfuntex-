"""Test remote Colab connection with 30s timeout."""
import sys
sys.path.insert(0, '.')
from colab.remote_executor import RemoteColabExecutor
import os

brave = r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'
bp = brave if os.path.isfile(brave) else None
print(f'Browser: {bp}', flush=True)
print(f'Session dir: {os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab", ".colab-session"))}', flush=True)

r = RemoteColabExecutor(headless=True, browser_path=bp)
result = r.connect(timeout=30)
print(f'Result: {result}', flush=True)
if result.get('success'):
    print('Connected!', flush=True)
    r.disconnect()
