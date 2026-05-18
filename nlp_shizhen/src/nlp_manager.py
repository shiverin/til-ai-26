"""Loads the models and runs the RAG QA pipeline.

NLPManager owns the models, builds the retrieval index when the corpus arrives
(load_corpus), and answers questions (qa / qa_batch).

Generation uses vLLM: continuous batching, PagedAttention and automatic prefix
caching make batched inference far faster than a per-call HF pipeline.
"""
import glob
import os
import re

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from vllm import LLM, SamplingParams

from chunking import chunk_corpus
from retrieval import Retriever

MODELS_DIR = "/app/models"
PHI_MODEL_ID = "microsoft/Phi-4-mini-instruct"
EMBEDDER_ID = "BAAI/bge-large-en-v1.5"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"

# Tunable pipeline constants.
MAX_CONTEXT_TOKENS = 1024   # top reranked chunks; small prompt = fast prefill
MAX_NEW_TOKENS = 256
# Fraction of GPU memory vLLM reserves (weights + KV cache). The rest holds the
# embedder + reranker (~1.2 GB) and CUDA overhead on the 15 GB T4.
GPU_MEM_FRACTION = 0.70
MAX_MODEL_LEN = 3072

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)

# Phi-4-mini sometimes emits stray spaces around digits/punctuation
# ("120, 000", "71 %", "3. 1", "korren - 8"). These repair the obvious cases.
# Comma/period spaces are collapsed only when digits flank BOTH sides, so an
# abbreviation space ("vs. 1.4%") is left intact.
_NUM_PUNCT = re.compile(r"(\d)\s*([,.])\s*(\d)")
_SP_PCT = re.compile(r"(\d)\s+%")
_SP_HYPHEN = re.compile(r"(\w)\s+-\s+(\w)")
_SOURCE_PREFIX = re.compile(r"^\s*Source\s+doc_?\d+\s*[:>]\s*", re.IGNORECASE)


def _normalize_answer(answer: str) -> str:
    """Repair stray spacing artifacts and strip echoed 'Source doc' labels."""
    answer = _SOURCE_PREFIX.sub("", answer)
    answer = _NUM_PUNCT.sub(r"\1\2\3", answer)
    answer = _NUM_PUNCT.sub(r"\1\2\3", answer)  # 2nd pass: chained groups
    answer = _SP_PCT.sub(r"\1%", answer)
    answer = _SP_HYPHEN.sub(r"\1-\2", answer)
    return answer.strip()


def _resolve_model_path(repo_id: str) -> str:
    """Return the local HF-cache snapshot directory for a repo, else the id."""
    cache = os.path.join(
        MODELS_DIR, "models--" + repo_id.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(cache, "*")))
    return snaps[-1] if snaps else repo_id


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
        "SOURCES:\nSource doc_0: Project Liminal had a total development cost "
        "of 12.4 billion Credits. Low-end projected annual recurring revenue "
        "is 40 billion Credits.\n\nQUESTION: How many years of post-launch "
        "revenue at the low end would recoup the development cost?")},
    {"role": "assistant", "content": (
        "12.4 billion / 40 billion per year = 0.31 years.\n"
        "FINAL ANSWER: less than one year")},
]


class NLPManager:
    """RAG QA pipeline: hybrid retrieval + vLLM (Phi-4-mini) generation."""

    loaded = False

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # vLLM is constructed first so it can reserve its memory pool; the
        # embedder + reranker then load into the remaining GPU memory.
        print(">>> NLPManager: loading vLLM generator", flush=True)
        # enforce_eager: skip CUDA-graph capture — frees ~1.5 GB of VRAM for the
        # KV cache (the T4 is memory-tight) and speeds engine startup. Continuous
        # batching + PagedAttention remain the dominant speed wins.
        self.llm = LLM(
            model=_resolve_model_path(PHI_MODEL_ID),
            dtype="float16",
            gpu_memory_utilization=GPU_MEM_FRACTION,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

        # Load the helper models directly in fp16 (via model_kwargs) rather than
        # fp32-then-.half() — the fp32 peak would OOM the VRAM-tight T4.
        print(">>> NLPManager: loading embedder + reranker", flush=True)
        fp16 = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self.embedder = SentenceTransformer(
            EMBEDDER_ID, cache_folder=MODELS_DIR, local_files_only=True,
            device=device, model_kwargs=fp16)
        self.reranker = CrossEncoder(
            RERANKER_ID, cache_folder=MODELS_DIR, local_files_only=True,
            device=device, model_kwargs=fp16)

        self.retriever = Retriever(self.embedder, self.reranker)
        self.doc_ids = []  # corpus-position -> document id, set by load_corpus
        print(">>> NLPManager: ready", flush=True)

    def load_corpus(self, documents):
        """Chunk the corpus and build the retrieval index.

        documents: list of {"id": str, "document": str} dicts (current task
        contract). Plain strings are also accepted for backward compatibility.
        """
        texts = []
        self.doc_ids = []
        for i, d in enumerate(documents):
            if isinstance(d, dict):
                texts.append(d.get("document", "") or "")
                self.doc_ids.append(d.get("id", f"doc_{i}"))
            else:
                texts.append(d)
                self.doc_ids.append(f"doc_{i}")
        chunks = chunk_corpus(texts, self.embedder.tokenizer)
        self.retriever.index(chunks)
        self.loaded = True
        print(f">>> NLPManager: indexed {len(chunks)} chunks "
              f"from {len(texts)} documents", flush=True)

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

    def _build_messages(self, question, chunks):
        """Assemble the chat prompt (system + few-shot + sources) for a question."""
        context = self._build_context(chunks)
        return (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + FEWSHOT
            + [{"role": "user", "content":
                f"SOURCES:\n{context}\n\nQUESTION: {question}"}]
        )

    @staticmethod
    def _parse_answer(text):
        """Extract and normalize the answer after the last 'FINAL ANSWER:' marker."""
        matches = list(_FINAL_ANSWER_RE.finditer(text))
        if matches:
            answer = matches[-1].group(1).strip()
            answer = answer.splitlines()[0].strip() if answer else ""
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            answer = lines[-1] if lines else ""
        return _normalize_answer(answer)

    def _retrieved_doc_ids(self, chunks):
        """Distinct parent-document ids of the chunks, in rerank order, top 3."""
        out = []
        for c in chunks:
            did = self.doc_ids[c.doc_index]
            if did not in out:
                out.append(did)
        return out[:3]

    def qa_batch(self, questions):
        """Answer a batch of questions in one vLLM continuous-batched pass.

        Returns a list of {"answer", "documents"} dicts aligned with questions.
        """
        if not self.loaded:
            return [{"answer": "", "documents": []} for _ in questions]

        # Retrieve for the whole batch at once (batched embed + rerank).
        try:
            retrieved = self.retriever.retrieve_batch(questions)
        except Exception as e:
            print(f">>> qa_batch retrieval error: {e}", flush=True)
            retrieved = [[] for _ in questions]

        results = [{"answer": "", "documents": self._retrieved_doc_ids(c)}
                   for c in retrieved]

        # Generate only for questions that retrieved context.
        gen_idx, convs = [], []
        for i, (q, chunks) in enumerate(zip(questions, retrieved)):
            if chunks:
                gen_idx.append(i)
                convs.append(self._build_messages(q, chunks))

        if convs:
            try:
                outputs = self.llm.chat(convs, self.sampling, use_tqdm=False)
                for i, out in zip(gen_idx, outputs):
                    text = out.outputs[0].text if out.outputs else ""
                    results[i]["answer"] = self._parse_answer(text)
            except Exception as e:  # keep retrieved docs for partial credit
                print(f">>> qa_batch generation error: {e}", flush=True)
        return results

    def qa(self, question):
        """Answer one question -> {"answer", "documents"}."""
        return self.qa_batch([question])[0]
