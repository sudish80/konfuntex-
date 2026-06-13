#!/usr/bin/env python3
"""Streamlit UI for Colab Agent."""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from config.settings import settings
from storage.database import init_db
from storage.jobs import JobStore
from storage.conversations import ConversationStore
from storage.models_store import ModelVersionStore
from agent.core import run_agent

st.set_page_config(page_title="Colab Agent", page_icon="🤖", layout="wide")

# Init DB
os.makedirs(settings.data_dir, exist_ok=True)
init_db()

st.title("🤖 Colab Agent")
st.markdown("Autonomous LLM-powered interface for fine-tuning models in Google Colab")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Agent Chat", "Jobs", "Models", "Conversations", "Colab Code", "Settings"])

    st.divider()
    st.markdown("### Quick Actions")
    if st.button("📊 List All Jobs"):
        page = "Jobs"
    if st.button("💬 New Chat"):
        page = "Agent Chat"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "job_store" not in st.session_state:
    st.session_state.job_store = JobStore()
if "conv_store" not in st.session_state:
    st.session_state.conv_store = ConversationStore()
if "model_store" not in st.session_state:
    st.session_state.model_store = ModelVersionStore()

if page == "Agent Chat":
    st.header("💬 Agent Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("What model do you want to fine-tune?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = run_agent(prompt)
                summary = result.get("summary", json.dumps(result, indent=2))
                st.markdown(summary)

                if result.get("job_id"):
                    st.success(f"Job ID: {result['job_id']}")

                with st.expander("Full Result"):
                    st.json(result)

        st.session_state.messages.append({"role": "assistant", "content": summary})

elif page == "Jobs":
    st.header("📋 Fine-tuning Jobs")

    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status Filter", ["All", "pending", "running", "completed", "failed"])
    with col2:
        method_filter = st.selectbox("Method Filter", ["All", "lora", "qlora", "full"])
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()

    jobs = st.session_state.job_store.list()
    if status_filter != "All":
        jobs = [j for j in jobs if j.status == status_filter]
    if method_filter != "All":
        jobs = [j for j in jobs if j.method == method_filter]

    if not jobs:
        st.info("No jobs found")
    else:
        for job in jobs[:50]:
            with st.expander(f"{job.id[:8]} - {job.goal[:80]} [{job.status}]"):
                st.markdown(f"**Goal:** {job.goal}")
                st.markdown(f"**Method:** {job.method or 'N/A'} | **Model:** {job.base_model or 'N/A'} | **Dataset:** {job.dataset or 'N/A'}")
                st.markdown(f"**Runtime:** {job.runtime or 'N/A'} | **Status:** {job.status}")
                st.markdown(f"**Created:** {job.created_at} | **Updated:** {job.updated_at}")
                if job.metrics:
                    st.markdown("**Metrics:**")
                    st.json(job.get_metrics())
                if job.error:
                    st.error(job.error)

elif page == "Models":
    st.header("🏗️ Model Versions")
    models = st.session_state.model_store.list_all()
    if not models:
        st.info("No model versions stored")
    else:
        for m in models[:30]:
            with st.expander(f"{m.base_model} ({m.method or 'N/A'}) - {m.id[:8]}"):
                st.markdown(f"**Base Model:** {m.base_model}")
                st.markdown(f"**Method:** {m.method or 'N/A'}")
                st.markdown(f"**Runtime:** {m.runtime_used or 'N/A'}")
                st.markdown(f"**Created:** {m.created_at}")
                if m.hf_repo_id:
                    st.markdown(f"**HF Hub:** [{m.hf_repo_id}](https://huggingface.co/{m.hf_repo_id})")
                if m.metrics:
                    st.json(m.get_metrics())

elif page == "Conversations":
    st.header("💬 Conversation History")
    convs = st.session_state.conv_store.list_all()
    if not convs:
        st.info("No conversations found")
    else:
        for conv in convs[:20]:
            with st.expander(f"{conv.goal[:80]} [{conv.status}] - {conv.id[:8]}"):
                st.markdown(f"**Goal:** {conv.goal}")
                st.markdown(f"**Status:** {conv.status}")
                st.markdown(f"**Messages:** {len(conv.get_messages())}")
                st.markdown(f"**Updated:** {conv.updated_at}")
                if conv.summary:
                    st.markdown(f"**Summary:** {conv.summary}")
                for msg in conv.get_messages()[-5:]:
                    st.markdown(f"**{msg['role']}:** {msg['content'][:300]}...")

elif page == "Colab Code":
    st.header("📓 Colab Setup Code")
    st.markdown("Copy and paste this into a new Google Colab notebook to get started.")

    code = '''# ===== Colab Agent Standalone Setup =====
!pip install -q transformers datasets accelerate peft trl bitsandbytes huggingface_hub torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# === CONFIG ===
MODEL_NAME = "microsoft/phi-2"
DATASET_NAME = "databricks/databricks-dolly-15k"
METHOD = "qlora"  # lora, qlora, or full
OUTPUT_DIR = "./finetuned_model"

# === LOAD MODEL ===
bnb_config = None
if METHOD == "qlora":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
)
print(f"Model loaded! Params: {model.num_parameters() / 1e9:.2f}B")

# === PEFT ===
if METHOD in ("lora", "qlora"):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

# === DATASET ===
dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset: {len(dataset)} samples")

# === TRAINING ===
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        logging_steps=25, save_steps=500, fp16=True,
        gradient_checkpointing=True, report_to="none",
    ),
    train_dataset=dataset.select(range(min(1000, len(dataset)))),
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
'''
    st.code(code, language="python")

    st.download_button(
        label="📥 Download as .ipynb",
        data=json.dumps({
            "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [{"cell_type": "code", "metadata": {"id": "setup"}, "source": [code], "outputs": []}]
        }),
        file_name="colab_agent_setup.ipynb",
        mime="application/json",
    )

elif page == "Settings":
    st.header("⚙️ Settings")
    st.markdown("Configure via environment variables or `.env` file.")

    config_items = {
        "COLAB_AGENT_LLM_PROVIDER": settings.llm_provider,
        "COLAB_AGENT_LLM_MODEL": settings.llm_model,
        "COLAB_AGENT_OPENAI_API_KEY": f"{settings.openai_api_key[:8]}..." if settings.openai_api_key else "",
        "COLAB_AGENT_HF_TOKEN": f"{settings.hf_token[:8]}..." if settings.hf_token else "",
        "COLAB_AGENT_GITHUB_TOKEN": f"{settings.github_token[:8]}..." if settings.github_token else "",
        "COLAB_AGENT_GITHUB_REPO": settings.github_repo or "",
        "COLAB_AGENT_DEFAULT_FINETUNE_METHOD": settings.default_finetune_method,
        "COLAB_AGENT_DEFAULT_BASE_MODEL": settings.default_base_model,
        "COLAB_AGENT_RUNTIME_AUTO_SWITCH": str(settings.colab_runtime_auto_switch),
        "Data Directory": settings.data_dir,
    }

    for key, val in config_items.items():
        st.text_input(key, value=val, disabled=True)

    st.markdown("---")
    st.markdown("**Runtime Tiers**")
    for tier, vram in settings.runtime_tiers.items():
        st.markdown(f"- **{tier}**: {vram}GB VRAM")

    st.markdown("---")
    if st.button("🔄 Reinitialize Database"):
        init_db()
        st.success("Database reinitialized!")
