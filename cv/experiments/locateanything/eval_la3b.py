"""Evaluate nvidia/LocateAnything-3B zero-shot on the TIL-26 CV val set.

Runs open-vocabulary detection over the 18 challenge classes, dumps COCO-format
detections, and scores mAP@.5:.95 with pycocotools against val_coco.json.

Usage:
    .venv/bin/python eval_la3b.py --limit 50 [--mode hybrid] [--dtype float16]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

CV_ROOT = Path(__file__).resolve().parents[2]
VAL_IMAGES = CV_ROOT / "finetune/data/val/images"
VAL_COCO = CV_ROOT / "finetune/data/val_coco.json"
OUT_DIR = Path(__file__).resolve().parent / "results"

CLASSES = [
    "cargo aircraft", "commercial aircraft", "drone", "fighter jet",
    "fighter plane", "helicopter", "light aircraft", "missile",
    "truck", "car", "tank", "bus", "van",
    "cargo ship", "yacht", "cruise ship", "warship", "sailboat",
]
NAME_TO_ID = {n: i for i, n in enumerate(CLASSES)}

DETECT_PROMPT = (
    "Locate all the instances that matches the following description: "
    + "</c>".join(CLASSES) + "."
)

BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def load_model(dtype):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    path = "nvidia/LocateAnything-3B"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
        .to("cuda")
        .eval()
    )
    return tok, proc, model


@torch.no_grad()
def predict(tok, proc, model, dtype, image, prompt, mode, max_new_tokens=2048):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(messages)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    response = model.generate(
        pixel_values=inputs["pixel_values"].to(dtype),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws"),
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        generation_mode=mode,
        do_sample=False,
        use_cache=True,
    )
    return response[0] if isinstance(response, tuple) else response


def parse_answer(answer, width, height):
    """Split the answer into label segments and attach following boxes."""
    dets = []
    pos = 0
    label = None
    for m in BOX_RE.finditer(answer):
        between = answer[pos:m.start()]
        cleaned = re.sub(r"<[^>]*>", " ", between).strip(" .,;:\n\t")
        if cleaned:
            label = cleaned.lower()
        pos = m.end()
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        cat = match_class(label)
        if cat is None:
            continue
        l = x1 / 1000 * width
        t = y1 / 1000 * height
        w = (x2 - x1) / 1000 * width
        h = (y2 - y1) / 1000 * height
        if w <= 0 or h <= 0:
            continue
        dets.append({"category_id": cat, "bbox": [l, t, w, h], "score": 1.0})
    return dets


def match_class(label):
    if label is None:
        return None
    if label in NAME_TO_ID:
        return NAME_TO_ID[label]
    for name, idx in NAME_TO_ID.items():
        if name in label:
            return idx
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--dump-raw", type=int, default=3, help="print raw answers for first N images")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    OUT_DIR.mkdir(exist_ok=True)

    gt = json.loads(VAL_COCO.read_text())
    images = gt["images"][: args.limit]
    print(f"Evaluating {len(images)} images, mode={args.mode}, dtype={args.dtype}")

    tok, proc, model = load_model(dtype)
    print("Model loaded.")

    detections, raws = [], []
    t0 = time.time()
    for i, info in enumerate(images):
        img = Image.open(VAL_IMAGES / info["file_name"]).convert("RGB")
        t1 = time.time()
        answer = predict(tok, proc, model, dtype, img, DETECT_PROMPT, args.mode)
        dt = time.time() - t1
        dets = parse_answer(answer, info["width"], info["height"])
        for d in dets:
            d["image_id"] = info["id"]
        detections.extend(dets)
        raws.append({"file": info["file_name"], "answer": answer, "secs": dt})
        if i < args.dump_raw:
            print(f"--- {info['file_name']} ({dt:.1f}s) ---\n{answer!r}\n  parsed: {len(dets)} dets")
        else:
            print(f"[{i+1}/{len(images)}] {info['file_name']}: {len(dets)} dets in {dt:.1f}s", flush=True)
    total = time.time() - t0
    print(f"\nTotal {total:.1f}s, {total/len(images):.2f}s/img")

    (OUT_DIR / f"dets_{args.mode}_{len(images)}.json").write_text(json.dumps(detections))
    (OUT_DIR / f"raw_{args.mode}_{len(images)}.json").write_text(json.dumps(raws, indent=1))

    if not detections:
        print("No detections parsed — inspect raw answers.")
        return

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(str(VAL_COCO))
    coco_subset_ids = [im["id"] for im in images]
    dt_coco = coco.loadRes(detections)
    ev = COCOeval(coco, dt_coco, "bbox")
    ev.params.imgIds = coco_subset_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"\nmAP@.5:.95 = {ev.stats[0]:.4f} on {len(images)} images")


if __name__ == "__main__":
    main()
