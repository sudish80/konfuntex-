import os
import pandas as pd
from datasets import Dataset
from agent.plugin import Plugin, plugin

@plugin(name="LocalDataIngestor", description="Handles CSV/JSON local data ingestion")
class LocalDataIngestor(Plugin):
    def before_code_gen(self, step: dict, prompt: str, context: dict) -> tuple[str, dict]:
        # If action is 'load_dataset' and file is local, inject ingestion code
        if step.get("action") == "load_dataset":
            # Check for local file hint in context
            local_path = context.get("data_path")
            if local_path and os.path.exists(local_path):
                ingestion_code = f"""
import pandas as pd
from datasets import Dataset
path = "{local_path}"
if path.endswith('.csv'): df = pd.read_csv(path)
else: df = pd.read_json(path)
dataset = Dataset.from_pandas(df)
print(f"Ingested {{len(dataset)}} samples from {{path}}")
"""
                return ingestion_code, context
        return prompt, context
