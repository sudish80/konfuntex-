import pandas as pd
import logging
from agent.plugin import Plugin, plugin

logger = logging.getLogger(__name__)

@plugin(name="DataCleaningPlugin", description="Automated cleaning for local datasets", priority=50)
class DataCleaningPlugin(Plugin):
    def before_code_gen(self, step: dict, code: str, context: dict) -> tuple[str, dict]:
        if step.get("action") == "load_dataset":
            # Inject cleaning logic into the script
            cleaning_code = """
# --- Automated Data Cleaning ---
df = df.dropna()
df = df.drop_duplicates()
# Basic text normalization
if 'text' in df.columns:
    df['text'] = df['text'].astype(str).str.strip()
print(f"Cleaned dataset: {len(df)} samples remaining")
# -------------------------------
"""
            return code.replace("dataset = Dataset.from_pandas(df)", cleaning_code + "\ndataset = Dataset.from_pandas(df)"), context
        return code, context
