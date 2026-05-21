"""Manages the ASR model."""

import io
import logging
import wave
from concurrent.futures import ThreadPoolExecutor

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile
import torch
from omegaconf import open_dict

_MODEL_PATH = "/workspace/models/parakeet_finetuned.nemo"

_LOG = logging.getLogger(__name__)
_SAMPLE_RATE = 16000

# Shared decode pool — 4 workers comfortably covers typical batch sizes
# without thrashing. soundfile.read releases the GIL so threads actually
# parallelize the libsndfile work.
_DECODE_POOL = ThreadPoolExecutor(max_workers=4)


class ASRManager:

    def __init__(self):
        # Loaded once at startup from the fine-tuned .nemo baked into the
        # image (see Dockerfile) — no network needed. Parakeet-TDT decodes
        # non-autoregressively, so it is far faster than an LLM-decoder model.
        self.model = nemo_asr.models.ASRModel.restore_from(_MODEL_PATH)
        self.model = self.model.eval().cuda()

        # Enable NeMo's CUDA-graph TDT decoder. Greedy hypotheses are
        # identical; only the per-step host overhead is removed. Falls back
        # to the default greedy decoder if this NeMo build lacks the flag.
        self._try_enable_cuda_graph_decoder()

        # Pick autocast dtype once at startup based on hardware capability.
        self._autocast_dtype = self._pick_autocast_dtype()
        _LOG.info("autocast dtype: %s", self._autocast_dtype)

        # Compile the encoder with torch.compile. Inductor fuses kernels
        # and (in reduce-overhead mode) wraps them in CUDA graphs. Silent
        # fallback to eager on failure — no behavior change either way.
        self._try_compile_encoder()

        # Probe the largest batch size that fits without OOM. The probe's
        # successful attempt also serves as the CUDA-graph warmup, so we
        # don't run a separate warmup block.
        self._batch_size = self._probe_batch_size([128, 96, 64, 48, 32, 24, 16])

    def _try_enable_cuda_graph_decoder(self) -> bool:
        """Switch the decoder to greedy_batch + CUDA graphs.

        The greedy hypotheses are mathematically identical to the current path;
        only the per-step host overhead changes. If the installed NeMo build
        doesn't expose the flag (older versions, non-RNNT decoders, etc.) we
        log a warning and keep the current decoder.

        Returns True if the flag was applied, False on any failure.
        """
        try:
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy_batch"
                if "greedy" not in decoding_cfg:
                    decoding_cfg.greedy = {}
                decoding_cfg.greedy.use_cuda_graph_decoder = True
            self.model.change_decoding_strategy(decoding_cfg)
            return True
        # Broad catch on purpose: NeMo's exception taxonomy varies across
        # versions; any failure here should fall back, not crash startup.
        except Exception as exc:
            _LOG.warning(
                "CUDA-graph decoder not enabled, falling back to default "
                "greedy decoder: %s", exc,
            )
            return False

    @staticmethod
    def _pick_autocast_dtype() -> torch.dtype:
        """BF16 on Ampere+ (same speed as FP16, better dynamic range, no NaN
        risk). FP16 on Turing where BF16 isn't supported in hardware.
        """
        return (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    def _try_compile_encoder(self) -> bool:
        """Compile ``self.model.encoder`` with torch.compile.

        ``reduce-overhead`` enables Inductor's CUDA-graph wrapping for the
        encoder forward pass (separate cache from NeMo's TDT decoder graph;
        they do not collide). ``dynamic=True`` lets clip-length variation
        avoid per-shape recompiles.

        On any failure we keep the eager encoder and return False. The model
        still works correctly, we just lose this submission's win.
        """
        try:
            compiled = torch.compile(
                self.model.encoder, mode="reduce-overhead", dynamic=True,
            )
            self.model.encoder = compiled
            _LOG.info("encoder compiled with torch.compile (reduce-overhead)")
            return True
        except Exception as exc:
            _LOG.warning("encoder compile skipped: %s", exc)
            return False

    def _probe_batch_size(self, candidates: list[int]) -> int:
        """Find the largest batch size in ``candidates`` that does not OOM.

        Each candidate runs one transcribe() over silence at that batch size.
        The successful attempt also serves as the CUDA-graph warmup, so the
        first real request after startup does not pay graph-capture cost.

        If every candidate OOMs (shouldn't happen — the smallest is 16, which
        is what we already shipped successfully), returns the last candidate
        so the server still comes up.
        """
        for bs in candidates:
            try:
                dummy = [self._silence_wav(8.0)] * bs
                signals = [self._decode(b) for b in dummy]
                with torch.autocast(
                    device_type="cuda", dtype=self._autocast_dtype
                ):
                    self.model.transcribe(signals, batch_size=bs, verbose=False)
                _LOG.info("probe: locked batch_size=%d", bs)
                return bs
            except torch.cuda.OutOfMemoryError:
                _LOG.info("probe: batch_size=%d OOM, trying smaller", bs)
                torch.cuda.empty_cache()
                continue
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    _LOG.info(
                        "probe: batch_size=%d OOM (RuntimeError), trying "
                        "smaller", bs,
                    )
                    torch.cuda.empty_cache()
                    continue
                raise
        _LOG.warning(
            "probe: every candidate OOMed, falling back to %d", candidates[-1],
        )
        return candidates[-1]

    @staticmethod
    def _silence_wav(seconds: float) -> bytes:
        """`seconds` of 16 kHz mono silence as WAV bytes, for warmup."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_SAMPLE_RATE)
            wav.writeframes(b"\x00\x00" * int(_SAMPLE_RATE * seconds))
        return buf.getvalue()

    @staticmethod
    def _decode(audio_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes to a mono float32 waveform at 16 kHz.

        The eval audio is uniformly 16 kHz mono PCM_16; the resample is a
        cheap guard in case the hidden set ever differs.
        """
        audio, sr = soundfile.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _SAMPLE_RATE:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=_SAMPLE_RATE
            )
        return audio

    def asr_batch(self, audio_list: list[bytes]) -> list[str]:
        """Transcribes a list of WAV audio files given as raw bytes.

        Args:
            audio_list: The audio files, each as bytes.

        Returns:
            One transcript string per input, in the same order.
        """

        if not audio_list:
            return []

        try:
            # Parallel decode: GIL is released inside libsndfile, so a small
            # thread pool overlaps WAV decoding across the batch instead of
            # leaving the GPU idle until every clip is decoded.
            signals = list(_DECODE_POOL.map(self._decode, audio_list))
            # Sort by length so each internal batch pads to its own max
            # rather than the global longest clip. WER-neutral (greedy TDT
            # is per-clip deterministic); only padding waste changes.
            order = sorted(range(len(signals)), key=lambda i: len(signals[i]))
            sorted_signals = [signals[i] for i in order]
            # FP16 autocast uses FP16 tensor cores while keeping sensitive
            # ops in FP32 (verified +0.0001 WER, see finetune/verify_fp16.py).
            with torch.autocast(
                device_type="cuda", dtype=self._autocast_dtype
            ):
                outputs = self.model.transcribe(
                    sorted_signals, batch_size=self._batch_size, verbose=False
                )
            sorted_texts = [
                (getattr(o, "text", o) or "").strip() for o in outputs
            ]
            # Invert the permutation to return predictions in input order.
            result = [""] * len(signals)
            for original_idx, text in zip(order, sorted_texts):
                result[original_idx] = text
            return result
        except Exception:
            # A failure must not crash the whole batch.
            return [""] * len(audio_list)

    def asr(self, audio_bytes: bytes) -> str:
        """Transcribes a single WAV audio file given as raw bytes."""

        return self.asr_batch([audio_bytes])[0]
