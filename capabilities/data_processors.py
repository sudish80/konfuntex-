"""
Phases 21-25 — Data Processing.

Provides:
  - SyntheticDataGenerator   generate instruction-response pairs  (21)
  - DataAugmenter            back-translation / EDA              (22)
  - PrivacyMasker            PII detection + redaction           (23)
  - ToxicityFilter           toxicity classification + removal   (24)
  - DeduplicationEngine      MinHash LSH near-dup detection      (25)
"""
import json
import os
from typing import Optional


# ==================================================================== #
#  21 — SyntheticDataGenerator
# ==================================================================== #

class SyntheticDataGenerator:
    """
    Generate diverse instruction-response pairs using an LLM.
    """

    TASK_TYPES = [
        "classification", "summarization", "code generation",
        "Q&A", "translation", "creative writing", "reasoning",
        "data extraction", "text completion", "explanation",
    ]

    def __init__(self, output_dir: str = "./synthetic_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_code(self, task_types: Optional[list[str]] = None,
                      num_per_task: int = 10,
                      model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
        tasks = task_types or self.TASK_TYPES
        tasks_json = json.dumps(tasks)
        template = r'''
import json, os, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM

output_dir = "__OUTPUT_DIR__"
os.makedirs(output_dir, exist_ok=True)

model_name = "__MODEL_NAME__"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)

task_types = __TASKS_JSON__
results = []

for task in task_types:
    prompt = f"""Generate __NUM_PER_TASK__ diverse {task} tasks.
For each, write:
- instruction: what to do
- input: context (optional)
- expected_output: correct answer
- difficulty: easy/medium/hard

Format as JSON array."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=2048, temperature=0.8, do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    import re
    examples = re.findall(
        r'{{"instruction".*?"expected_output".*?}}',
        response, re.DOTALL
    )
    for ex_str in examples:
        try:
            ex = json.loads(ex_str)
            ex["task_type"] = task
            ex["generator_model"] = model_name
            ex["temperature"] = 0.8
            results.append(ex)
        except json.JSONDecodeError:
            pass

    print(f"Task {task}: generated {len(examples)} examples")
    time.sleep(1)

with open(os.path.join(output_dir, "synthetic_data.jsonl"), "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\\n")

print(f"Total generated: {len(results)} examples")
'''
        code = template.replace("__OUTPUT_DIR__", self.output_dir)
        code = code.replace("__MODEL_NAME__", model_name)
        code = code.replace("__TASKS_JSON__", tasks_json)
        code = code.replace("__NUM_PER_TASK__", str(num_per_task))
        return code


# ==================================================================== #
#  22 — DataAugmenter
# ==================================================================== #

class DataAugmenter:
    """
    Augment text data via back-translation, synonym replacement,
    random insertion, and random deletion.
    """

    METHODS = ["back_translate", "synonym_replace", "random_insert",
               "random_delete"]

    def __init__(self, output_dir: str = "./augmented_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def augment_code(self, dataset_path: str = "./data/train.parquet",
                     text_column: str = "text",
                     methods: Optional[list[str]] = None,
                     augment_factor: int = 2) -> str:
        methods = methods or self.METHODS
        return f"""
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

df = pd.read_parquet("{dataset_path}")
texts = df["{text_column}"].tolist()
augmented_texts = []
augmented_indices = []

methods = {json.dumps(methods)}

# Synonym replacement (WordNet)
if "synonym_replace" in methods:
    aug_syn = naw.SynonymAug(aug_src="wordnet")
    for i, t in enumerate(texts):
        if len(t.split()) > 3:
            try:
                aug_texts.append({{"text": aug_syn.augment(t), "method": "synonym", "original_idx": i}})
            except Exception:
                pass

# Random insertion
if "random_insert" in methods:
    aug_ins = naw.RandomWordAug(action="insert")
    for i, t in enumerate(texts):
        if len(t.split()) > 3:
            try:
                aug_texts.append({{"text": aug_ins.augment(t), "method": "insert", "original_idx": i}})
            except Exception:
                pass

# Random deletion
if "random_delete" in methods:
    aug_del = naw.RandomWordAug(action="delete")
    for i, t in enumerate(texts):
        if len(t.split()) > 5:
            try:
                aug_texts.append({{"text": aug_del.augment(t), "method": "delete", "original_idx": i}})
            except Exception:
                pass

# Back-translation using Helsinki-NLP
if "back_translate" in methods:
    from transformers import pipeline
    en_to_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
    de_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")
    for i, t in enumerate(texts[:50]):  # Limit for speed
        try:
            de = en_to_de(t[:512])[0]["translation_text"]
            en = de_to_en(de)[0]["translation_text"]
            aug_texts.append({{"text": en, "method": "back_translate", "original_idx": i}})
        except Exception:
            pass

# Combine with original
aug_df = pd.DataFrame(aug_texts)
combined = pd.concat([df, aug_df], ignore_index=True)
print(f"Original: {{len(df)}}, Augmented: {{len(aug_df)}}, Total: {{len(combined)}}")
combined.to_parquet("{self.output_dir}/augmented_train.parquet")
"""


# ==================================================================== #
#  23 — PrivacyMasker
# ==================================================================== #

class PrivacyMasker:
    """
    Detect and redact PII from text using Microsoft Presidio.
    """

    ENTITY_TYPES = [
        "EMAIL", "PHONE_NUMBER", "CREDIT_CARD", "IP_ADDRESS",
        "PERSON", "LOCATION", "DATE_TIME", "NRP", "URL",
    ]

    def __init__(self, output_dir: str = "./privacy_masked"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def mask_code(self, dataset_path: str = "./data/train.parquet",
                  text_column: str = "text",
                  entity_types: Optional[list[str]] = None) -> str:
        entities = entity_types or self.ENTITY_TYPES
        return f"""
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

df = pd.read_parquet("{dataset_path}")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

entity_types = {json.dumps(entities)}
redaction_log = []

def mask_text(text: str, idx: int) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    results = analyzer.analyze(text, language="en", entities=entity_types)
    if results:
        redaction_log.append({{"idx": idx, "count": len(results),
                               "types": [r.entity_type for r in results]}})
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={{"DEFAULT": {{"type": "replace", "new_value": "[REDACTED_{{entity_type}}]"}}}},
    )
    return anonymized.text

df["masked_text"] = [mask_text(t, i) for i, t in enumerate(df["{text_column}"].tolist())]

# Report
log_df = pd.DataFrame(redaction_log)
print(f"Redacted {{len(log_df)}} / {{len(df)}} samples")
print(f"Total redactions: {{log_df['count'].sum() if len(log_df) else 0}}")
print(f"Entity types: {{log_df['types'].explode().value_counts().to_dict() if len(log_df) else {{}}}}")

df.to_parquet("{self.output_dir}/masked_dataset.parquet")
log_df.to_csv("{self.output_dir}/redaction_log.csv", index=False)
"""

    @staticmethod
    def custom_regex_code() -> str:
        return """
# Additional custom patterns for API keys, tokens, etc.
import re

CUSTOM_PATTERNS = {
    "API_KEY": r"sk-[a-zA-Z0-9_]{20,}",
    "GITHUB_TOKEN": r"gh[pousr]_[a-zA-Z0-9_]{36,}",
    "AWS_KEY": r"AKIA[0-9A-Z]{16}",
    "JWT": r"eyJ[a-zA-Z0-9_-]{10,}\\.[a-zA-Z0-9_-]{10,}\\.[a-zA-Z0-9_-]{10,}",
}

def apply_custom_patterns(text: str) -> str:
    for name, pattern in CUSTOM_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{name}]", text)
    return text
"""


# ==================================================================== #
#  24 — ToxicityFilter
# ==================================================================== #

class ToxicityFilter:
    """
    Filter toxic content using Detoxify multilingual toxicity model.
    """

    CATEGORIES = ["toxicity", "severe_toxicity", "obscene",
                  "threat", "insult", "identity_attack"]

    def __init__(self, output_dir: str = "./filtered_data",
                 threshold: float = 0.7):
        self.output_dir = output_dir
        self.threshold = threshold
        os.makedirs(output_dir, exist_ok=True)

    def filter_code(self, dataset_path: str = "./data/train.parquet",
                    text_column: str = "text",
                    threshold: float = 0.7) -> str:
        return f"""
import pandas as pd
from detoxify import Detoxify

df = pd.read_parquet("{dataset_path}")
model = Detoxify("multilingual")

threshold = {threshold}
keep_mask = []
filtered_count = 0

for i, text in enumerate(df["{text_column}"].tolist()):
    if not isinstance(text, str) or not text.strip():
        keep_mask.append(False)
        continue
    try:
        results = model.predict(text)
        max_score = max(results.get(cat, 0) for cat in {json.dumps(self.CATEGORIES)})
        if max_score >= threshold:
            filtered_count += 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    except Exception:
        keep_mask.append(True)

df_filtered = df[keep_mask].copy()
print(f"Filtered {{filtered_count}} toxic samples ({{len(df)}} -> {{len(df_filtered)}})")
df_filtered.to_parquet("{self.output_dir}/filtered_dataset.parquet")

# Log rejected
df_rejected = df[~pd.Series(keep_mask)]
if len(df_rejected):
    df_rejected.to_csv("{self.output_dir}/rejected_toxic.csv", index=False)
"""


# ==================================================================== #
#  25 — DeduplicationEngine
# ==================================================================== #

class DeduplicationEngine:
    """
    Near-duplicate detection using MinHash + LSH.
    """

    def __init__(self, output_dir: str = "./deduped_data",
                 threshold: float = 0.85,
                 num_perm: int = 128):
        self.output_dir = output_dir
        self.threshold = threshold
        self.num_perm = num_perm
        os.makedirs(output_dir, exist_ok=True)

    def dedup_code(self, dataset_path: str = "./data/train.parquet",
                   text_column: str = "text",
                   shingle_size: int = 5) -> str:
        return f"""
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH

df = pd.read_parquet("{dataset_path}")
texts = df["{text_column}"].tolist()

threshold = {self.threshold}
num_perm = {self.num_perm}
shingle_size = {shingle_size}

lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
duplicates = []
kept_indices = []

for i, text in enumerate(texts):
    if not isinstance(text, str) or not text.strip():
        kept_indices.append(i)
        continue

    text_lower = text.lower()
    shingles = {{text_lower[j:j+shingle_size] for j in range(len(text_lower) - shingle_size + 1)}}

    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))

    result = lsh.query(m)
    if not result:
        lsh.insert(f"doc_{{i}}", m)
        kept_indices.append(i)
    else:
        duplicates.append({{"idx": i, "duplicate_of": result[0], "text_preview": text[:80]}})

df_dedup = df.iloc[kept_indices]
print(f"Dedup: {{len(df)}} -> {{len(df_dedup)}} (removed {{len(duplicates)}})")

df_dedup.to_parquet("{self.output_dir}/deduped_dataset.parquet")

# Export duplicates for review
dup_df = pd.DataFrame(duplicates)
dup_df.to_csv("{self.output_dir}/duplicates_report.csv", index=False)
print(f"Duplicates report saved to {{self.output_dir}}")
"""
