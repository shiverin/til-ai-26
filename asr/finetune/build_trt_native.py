"""Export the Parakeet encoder to ONNX, then compile a native TensorRT engine.

Produces ``encoder.trt`` in the format ``asr_manager._TRTEncoder`` consumes
(loaded via ``trt.Runtime.deserialize_cuda_engine``). The existing
``build_trt_encoder.py`` uses ``torch_tensorrt`` and outputs a TorchScript
``.ts`` file, which the current runtime does not load — this is the
matching-path script.

Run inside a CUDA container:
  docker run --rm --gpus all --shm-size=8g \\
    -v /home/jupyter/til-ai-26/asr/finetune:/work \\
    -v /home/jupyter/til-ai-26/asr/models:/models:ro \\
    nobrainnohack-asr:latest python /work/build_trt_native.py
"""

import os
import subprocess

import nemo.collections.asr as nemo_asr

MODEL = "/models/parakeet_finetuned.nemo"
ONNX = "/work/output/encoder.onnx"
TRT = "/work/output/encoder.trt"

# Dynamic-shape envelope. The runtime probe goes up to batch 128 and the
# data manifest has clips up to ~40 s, so the engine must cover that.
#   audio_signal: [batch, n_mels, time_steps]  (n_mels fixed per model)
#   length:       [batch]
# We auto-detect n_mels from the loaded model so the shapes can't drift.
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 16, 32      # was 128 — that requested 14 GB
MIN_T, OPT_T, MAX_T = 10, 800, 2400              # was 4000 — 24 s max is plenty


def main() -> None:
    # CPU-only export so the full GPU is available to trtexec afterwards.
    print(f"loading {MODEL} on CPU ...", flush=True)
    model = nemo_asr.models.ASRModel.restore_from(MODEL).eval()
    n_mels = int(model.cfg.preprocessor.features)
    print(f"  preprocessor features (n_mels) = {n_mels}", flush=True)

    print(f"exporting encoder -> {ONNX} ...", flush=True)
    os.makedirs(os.path.dirname(ONNX), exist_ok=True)
    model.encoder.export(ONNX)
    print(f"  ONNX written ({os.path.getsize(ONNX) // 1024 // 1024} MB)",
          flush=True)
    del model  # free the NeMo objects so trtexec gets all available RAM

    min_shape = f"audio_signal:{MIN_BATCH}x{n_mels}x{MIN_T},length:{MIN_BATCH}"
    opt_shape = f"audio_signal:{OPT_BATCH}x{n_mels}x{OPT_T},length:{OPT_BATCH}"
    max_shape = f"audio_signal:{MAX_BATCH}x{n_mels}x{MAX_T},length:{MAX_BATCH}"

    cmd = [
        "trtexec",
        f"--onnx={ONNX}",
        f"--saveEngine={TRT}",
        "--fp16",
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
        "--memPoolSize=workspace:4096",
    ]
    print("building TRT engine:", flush=True)
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    print(f"DONE: {TRT} "
          f"({os.path.getsize(TRT) // 1024 // 1024} MB)", flush=True)


if __name__ == "__main__":
    main()
