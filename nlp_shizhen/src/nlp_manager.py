"""Loads the models and runs the RAG QA pipeline.

NLPManager owns the three models, builds the retrieval index when the corpus
arrives (load_corpus), and answers questions (qa).
"""
import re

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from chunking import chunk_corpus
from retrieval import Retriever

MODELS_DIR = "/app/models"
PHI_MODEL_ID = "microsoft/Phi-4-mini-instruct"
EMBEDDER_ID = "BAAI/bge-large-en-v1.5"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"

# Tunable pipeline constants (adjusted in Task 11 using eval_local.py).
MAX_CONTEXT_TOKENS = 2800
MAX_NEW_TOKENS = 256
# Return "" only if the best reranked chunk scores below this. Default 0.0
# disables the empty-answer path (all local questions are answerable).
EMPTY_ANSWER_RERANK_THRESHOLD = 0.0

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "You answer questions about the fictional world of Clairos using ONLY the "
    "provided SOURCES.\n"
    "Rules:\n"
    "1. Answer with the shortest phrase that fully answers the question — "
    "usually 1 to 8 words. Never write a full sentence and never restate the "
    "question.\n"
    "2. Copy names, acronyms, codenames, dates, codes, numbers and their "
    "capitalization EXACTLY as written in the SOURCES. Do not lowercase, "
    "rewrite, expand, round, or convert them (e.g. keep '78 PCE' as '78 PCE', "
    "keep 'SEASTITCH' uppercase).\n"
    "3. If the question needs arithmetic or multi-step reasoning, work through "
    "it step by step first, then give the final result.\n"
    "4. Do not add explanations or phrases like 'According to the sources'.\n"
    "5. End your reply with a line in exactly this format:\n"
    "FINAL ANSWER: <shortest answer>"
)

# Few-shot turns teach the answer format: ultra-terse, exact casing preserved,
# and that reasoning questions are computed step by step before the result.
FEWSHOT = [
    {"role": "user", "content": (
        "SOURCES:\nSource doc_0: ONE Network Enterprises referred to the "
        "classified data-sharing arrangement internally by the codename "
        "SEASTITCH.\n\nQUESTION: What internal codename did ONE Network "
        "Enterprises use for the arrangement?")},
    {"role": "assistant", "content": "FINAL ANSWER: SEASTITCH"},
    {"role": "user", "content": (
        "SOURCES:\nSource doc_0: The Division assessed a penalty of 4.5 "
        "million Credits and mandatory equipment surrender against the "
        "respondent.\n\nQUESTION: What penalty was assessed against the "
        "respondent?")},
    {"role": "assistant", "content": (
        "FINAL ANSWER: 4.5 million Credits and mandatory equipment surrender")},
    {"role": "user", "content": (
        "SOURCES:\nSource doc_0: Project Liminal had a total development cost "
        "of 12.4 billion Credits. Low-end projected annual recurring revenue "
        "is 40 billion Credits.\n\nQUESTION: How many years of post-launch "
        "revenue at the low end would recoup the development cost?")},
    {"role": "assistant", "content": (
        "12.4 billion / 40 billion per year = 0.31 years.\n"
        "FINAL ANSWER: less than one year")},
]


class NLPManager:
    """RAG QA pipeline: retrieval + Phi-4-mini generation."""

    loaded = False

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(">>> NLPManager: loading generator", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            PHI_MODEL_ID, cache_dir=MODELS_DIR, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            PHI_MODEL_ID, cache_dir=MODELS_DIR, local_files_only=True,
            device_map="auto", torch_dtype=torch.float16,
            attn_implementation="sdpa")
        self.pipe = pipeline("text-generation", model=self.model,
                             tokenizer=self.tokenizer)

        print(">>> NLPManager: loading embedder + reranker", flush=True)
        self.embedder = SentenceTransformer(
            EMBEDDER_ID, cache_folder=MODELS_DIR, local_files_only=True,
            device=device)
        self.reranker = CrossEncoder(
            RERANKER_ID, cache_folder=MODELS_DIR, local_files_only=True,
            device=device)
        # fp16 for the helper models keeps total VRAM within the T4's 15 GB.
        if device == "cuda":
            self.embedder.half()
            self.reranker.model.half()

        self.retriever = Retriever(self.embedder, self.reranker)
        print(">>> NLPManager: ready", flush=True)

    def load_corpus(self, documents):
        """Chunk the corpus and build the retrieval index."""
        chunks = chunk_corpus(documents, self.embedder.tokenizer)
        self.retriever.index(chunks)
        self.loaded = True
        print(f">>> NLPManager: indexed {len(chunks)} chunks "
              f"from {len(documents)} documents", flush=True)

    def _build_context(self, chunks):
        """Join reranked chunks into a context string within the token budget."""
        parts, used = [], 0
        for c in chunks:
            ids = self.tokenizer(c.text, add_special_tokens=False)["input_ids"]
            if used + len(ids) > MAX_CONTEXT_TOKENS:
                ids = ids[:MAX_CONTEXT_TOKENS - used]
                if ids:
                    parts.append(self.tokenizer.decode(
                        ids, skip_special_tokens=True))
                break
            parts.append(c.text)
            used += len(ids)
        return "\n\n".join(f"Source doc_{i}: {t}" for i, t in enumerate(parts))

    @staticmethod
    def _parse_answer(text):
        """Extract the answer after the last 'FINAL ANSWER:' marker."""
        matches = list(_FINAL_ANSWER_RE.finditer(text))
        if matches:
            answer = matches[-1].group(1).strip()
            # keep only the first line of whatever followed the marker
            return answer.splitlines()[0].strip() if answer else ""
        # fallback: last non-empty line of the model output
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    def qa(self, question):
        """Answer one question. Returns "" on retrieval miss or any failure."""
        if not self.loaded:
            return ""
        try:
            chunks = self.retriever.retrieve(question)
            if not chunks:
                return ""
            if (self.retriever.last_top_score is not None
                    and self.retriever.last_top_score
                    < EMPTY_ANSWER_RERANK_THRESHOLD):
                return ""

            context = self._build_context(chunks)
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + FEWSHOT
                + [{"role": "user", "content":
                    f"SOURCES:\n{context}\n\nQUESTION: {question}"}]
            )
            output = self.pipe(
                messages, max_new_tokens=MAX_NEW_TOKENS,
                return_full_text=False, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generated = output[0].get("generated_text", "") if output else ""
            if isinstance(generated, list):
                last = generated[-1] if generated else {}
                generated = (last.get("content", "")
                             if isinstance(last, dict) else str(last))
            return self._parse_answer(str(generated))
        except Exception as e:  # one bad question must not abort the batch
            print(f">>> NLPManager.qa error: {e}", flush=True)
            return ""
