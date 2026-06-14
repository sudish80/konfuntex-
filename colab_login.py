"""One-time Colab Google login — opens browser, saves session."""
import sys
import os
sys.path.insert(0, '.')
from colab.remote_executor import RemoteColabExecutor

brave_candidates = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
]
browser_path = next((p for p in brave_candidates if os.path.isfile(p)), None)
if browser_path:
    print(f"Using Brave: {browser_path}")
else:
    print("Brave not found, using bundled Chromium")

r = RemoteColabExecutor(browser_path=browser_path)
success = r.login_once()
print(f"Login {'succeeded' if success else 'failed'}")
input("Press any key to exit...")
