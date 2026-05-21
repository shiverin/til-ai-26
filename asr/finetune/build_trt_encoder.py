"""Compile the Parakeet encoder to a TensorRT engine, save as TorchScript.

Run once on the host (needs a free T4). Output `asr/models/encoder_trt.ts`
is COPIED into the Docker image and loaded at container startup by
ASRManager._try_load_trt_encoder.
"""

import sys

import torch
import torch_tensorrt
import nemo.collections.asr as nemo_asr

NEMO_PATH = "asr/models/parakeet_finetuned.nemo"
OUTPUT_PATH = "asr/models/encoder_trt.ts"

# Dynamic-shape profile. min/opt/max for (audio_signal, length).
# audio_signal: (B, 80, T) — mel features, 80 bins, ~100 fps
# length:       (B,) int64 frame counts
SIGNAL_MIN = (1, 80, 80)        # ~0.8 s, batch 1
SIGNAL_OPT = (32, 80, 800)      # 8 s, batch 32 (typical eval shape)
SIGNAL_MAX = (128, 80, 2400)    # 24 s, batch 128 (worst case)
LENGTH_MIN = (1,)
LENGTH_OPT = (32,)
LENGTH_MAX = (128,)


def main() -> int:
    print(f"loading {NEMO_PATH}")
    model = nemo_asr.models.ASRModel.restore_from(NEMO_PATH).eval().cuda()
    encoder = model.encoder

    inputs = [
        torch_tensorrt.Input(
            min_shape=SIGNAL_MIN, opt_shape=SIGNAL_OPT, max_shape=SIGNAL_MAX,
            dtype=torch.float32,
        ),
        torch_tensorrt.Input(
            min_shape=LENGTH_MIN, opt_shape=LENGTH_OPT, max_shape=LENGTH_MAX,
            dtype=torch.int64,
        ),
    ]

    print("compiling with torch_tensorrt (this can take 5-15 min)")
    try:
        trt_encoder = torch_tensorrt.compile(
            encoder,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
            truncate_long_and_double=True,
        )
    except Exception as exc:
        print(f"dynamo IR failed ({exc}); retrying with ts IR")
        trt_encoder = torch_tensorrt.compile(
            encoder,
            ir="ts",
            inputs=inputs,
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
            truncate_long_and_double=True,
        )

    print(f"saving to {OUTPUT_PATH}")
    torch_tensorrt.save(trt_encoder, OUTPUT_PATH)

    # Quick sanity-check at the opt shape
    print("sanity check: forward at opt shape")
    sig = torch.randn(*SIGNAL_OPT, device="cuda", dtype=torch.float32)
    length = torch.full(LENGTH_OPT, SIGNAL_OPT[2],
                         device="cuda", dtype=torch.int64)
    with torch.no_grad():
        eager_out = encoder(sig, length)
        trt_out = trt_encoder(sig, length)
    if isinstance(eager_out, tuple):
        for i, (e, t) in enumerate(zip(eager_out, trt_out)):
            diff = (e.float() - t.float()).abs().max().item()
            print(f"  out[{i}] max abs diff: {diff:.6f}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
