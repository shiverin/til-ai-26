"""RAFT fine-tuning: LoRA SFT of Phi-4-mini on retrieval-grounded examples.

Trains a LoRA adapter so the generator learns to read the retrieved SOURCES
and emit the terse, exact-copy answer format the scorer rewards. The base model
stays frozen; only the adapter is trained, so it is small and composable, and
vLLM can serve it via enable_lora.

fp16 LoRA (not QLoRA): the 3.8B base in fp16 (~7.6 GB) + LoRA params + gradient
checkpointing fits a 15 GB T4 without bitsandbytes — one less fragile dep.

Loss is computed on the assistant turn only (assistant_only_loss) — the model
is graded on the answer, not on reproducing the prompt.

Usage:
  python raft_train.py            # writes models/raft_adapter/
"""
import glob
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager as nm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "raft_data", "train.jsonl")
OUT = os.path.join(HERE, "models", "raft_adapter")


def main():
    nm.MODELS_DIR = os.path.join(HERE, "models")
    base_path = nm._resolve_model_path(nm.PHI_MODEL_ID)

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="cuda")
    model.config.use_cache = False  # required with gradient checkpointing

    # LoRA on attention + MLP projections — the standard high-coverage target
    # set; r=16 is enough for a format/extraction-skill adapter on 733 examples.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )

    # Convert messages dataset to prompt/completion. trl masks the prompt and
    # trains only on the completion — same effect as assistant_only_loss but
    # without needing {% generation %} markers in the chat template (which
    # Phi-4-mini's template lacks).
    raw = load_dataset("json", data_files=DATA, split="train")
    def to_pc(ex):
        prompt = tok.apply_chat_template(
            ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "completion": ex["messages"][-1]["content"]}
    dataset = raw.map(to_pc, remove_columns=raw.column_names)
    print(f">>> {len(dataset)} training examples (prompt/completion format)",
          flush=True)

    cfg = SFTConfig(
        output_dir=OUT,
        num_train_epochs=2,
        per_device_train_batch_size=1,   # vocab ~200k -> shift_logits is big;
        gradient_accumulation_steps=16,  # batch=1 keeps it in VRAM. effective 16.
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        max_length=2048,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=dataset,
        peft_config=lora, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print(f">>> LoRA adapter saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
