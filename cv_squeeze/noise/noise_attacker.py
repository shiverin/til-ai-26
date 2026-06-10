"""Failure-mode-targeted adversarial noise attacker.

Three composable transforms, each designed to excite a specific CAM-identified
failure mode in YOLO detectors trained on the TIL composite dataset:

  add_fake_halo     → L-15 backbone frame-edge attention
  add_sky_noise     → L-8 P3 small-object head firing on sky/water texture
  add_fake_horizon  → L-2 semantic head horizon-line bias

Pure numpy + cv2. No torch, no GPU, deterministic given a seed.

Designed dual-use:
  - Training-time aug: strength ~0.7-1.0, no SSIM budget (~3-5× deployment)
  - Deployment noise service: strength ~0.3-0.5, tuned to pass SSIM/RMSE gate

See cv_squeeze/docs/adversarial-noise-aug-design.md for the strategy.
"""
import cv2  # type: ignore
import numpy as np


def add_fake_halo(img: np.ndarray, rng: np.random.Generator,
                  n: int = 3, thickness_range: tuple = (4, 10)) -> np.ndarray:
    """Draw n bright-or-dark rectangular outlines at random positions.

    Each rectangle spans 10–25% of image dimensions. Mixed bright/dark so the
    backbone can't shortcut to a single color. These mimic the matte-edge
    halos around composited objects in training data — exciting the backbone's
    "frame-edge = object" overfit at locations where there is no object.
    """
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(n):
        rw = int(rng.integers(int(w * 0.1), int(w * 0.25)))
        rh = int(rng.integers(int(h * 0.1), int(h * 0.25)))
        x = int(rng.integers(0, w - rw))
        y = int(rng.integers(0, h - rh))
        thickness = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
        # 50/50 bright vs dark
        c = 255 if rng.random() > 0.5 else 0
        cv2.rectangle(out, (x, y), (x + rw, y + rh), (c, c, c), thickness)
    return out


def add_sky_noise(img: np.ndarray, rng: np.random.Generator,
                  var_threshold: int = 50,
                  amplitude_range: tuple = (15, 40)) -> np.ndarray:
    """Inject high-frequency Gaussian noise into low-local-variance regions.

    Detects "solid-color" regions (sky, water, wall) by local variance
    against a Gaussian-blurred reference, then adds gaussian noise ONLY in
    those regions. This is where the L-8 small-object head was already
    firing on natural texture — adding more texture saturates it with
    false-positive activations.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    local_var = np.abs(gray - blurred)
    mask = (local_var < var_threshold).astype(np.float32)  # 1 = solid-color region
    amplitude = float(rng.integers(amplitude_range[0], amplitude_range[1] + 1))
    noise = rng.normal(0.0, amplitude, (h, w)).astype(np.float32)
    noise_3ch = np.stack([noise, noise, noise], axis=-1)
    mask_3ch = mask[..., None]
    out = img.astype(np.float32) + noise_3ch * mask_3ch
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def blur_hard_edges(img: np.ndarray, rng: np.random.Generator,
                    canny_low: int = 80, canny_high: int = 180,
                    dilate_px: int = 11, blur_ksize: int = 41) -> np.ndarray:
    """Edge-aware blur: detect harsh edges, soften only those regions.

    Targets the actual L-15 failure mode — the matte-halo ring around
    composite objects. Canny picks up the halos (and other real edges);
    a small dilate widens the mask to cover the full ring; Gaussian blur
    softens the masked pixels. Smooth regions (sky, body interiors) are
    untouched.

    If the model was depending on matte-edge cues to detect objects, this
    should cause detections to drop. Pure edge-preserving smoothing in the
    inverse sense.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        edges = cv2.dilate(edges, kernel)
    mask = (edges > 0).astype(np.float32)[..., None]  # (h, w, 1)
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    out = img.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    return out.astype(np.uint8)


def blur_object_edges(img: np.ndarray, boxes_xyxy: np.ndarray,
                      canny_low: int = 80, canny_high: int = 180,
                      pad_px: int = 11, dilate_px: int = 11,
                      blur_ksize: int = 41,
                      mask_feather_px: int = 15) -> np.ndarray:
    """Blur only the edges INSIDE (+padded) ground-truth bounding boxes.

    Surgical version of `blur_hard_edges`: instead of blurring every Canny
    edge in the image (including trees, hills, horizon), restrict to the
    region near object boxes. This matches the L-15 matte-halo bug exactly —
    the harsh ring exists *around objects*, nowhere else.

    boxes_xyxy: (N, 4) ground-truth boxes in pixel coords (x1, y1, x2, y2).
    pad_px: pad each box outward so the mask covers the matte ring just
        outside the visible object.
    mask_feather_px: Gaussian-blur the mask itself before compositing so
        the blurred region fades into untouched pixels with no visible seam.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - pad_px)
        y1 = max(0, int(y1) - pad_px)
        x2 = min(w, int(x2) + pad_px)
        y2 = min(h, int(y2) + pad_px)
        box_mask[y1:y2, x1:x2] = 255

    edges_in = cv2.bitwise_and(edges, box_mask)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    edges_dil = cv2.dilate(edges_in, kernel)

    # Feather the mask: smooth alpha blend, no hard seam at the blur boundary
    feather_k = mask_feather_px * 2 + 1
    soft_mask = cv2.GaussianBlur(
        edges_dil.astype(np.float32), (feather_k, feather_k), 0,
    ) / 255.0
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    out = (img.astype(np.float32) * (1.0 - soft_mask)
           + blurred.astype(np.float32) * soft_mask)
    return out.astype(np.uint8)


def soften_object_outline(img: np.ndarray, boxes_xyxy: np.ndarray,
                          canny_low: int = 80, canny_high: int = 180,
                          band_px: int = 8,
                          dilate_px: int = 3,
                          blur_ksize: int = 11,
                          feather_px: int = 7) -> np.ndarray:
    """Soften the real edges of objects that touch the bounding box.

    Method A: find Canny edges in the image, restrict them to a band along
    the box perimeter (so we get the silhouette where it meets the box,
    not interior edges or far-away edges), then apply a soft gradient
    blend with the locally-blurred image. Mimics natural photographic
    edge transitions (lens DOF, atmospheric haze) instead of the sharp
    matte cut of composited training data.

    band_px:   half-width of the perimeter band (in or out of the box)
    dilate_px: widen each matched edge for a thicker silhouette region
    blur_ksize: ksize of the local blur used to soften the matched edges
    feather_px: Gaussian-blur on the alpha mask itself for smooth gradient
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Band along the box perimeter: outer fill minus inner fill.
    band_mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        x1o = max(0, int(x1) - band_px)
        y1o = max(0, int(y1) - band_px)
        x2o = min(w, int(x2) + band_px)
        y2o = min(h, int(y2) + band_px)
        x1i = max(0, int(x1) + band_px)
        y1i = max(0, int(y1) + band_px)
        x2i = min(w, int(x2) - band_px)
        y2i = min(h, int(y2) - band_px)
        per_box = np.zeros((h, w), dtype=np.uint8)
        per_box[y1o:y2o, x1o:x2o] = 255
        if x2i > x1i and y2i > y1i:
            per_box[y1i:y2i, x1i:x2i] = 0
        band_mask = cv2.bitwise_or(band_mask, per_box)

    # Only the Canny edges that fall inside the perimeter band.
    edges_in_band = cv2.bitwise_and(edges, band_mask)
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        edges_in_band = cv2.dilate(edges_in_band, kernel)

    feather_k = feather_px * 2 + 1
    soft_alpha = cv2.GaussianBlur(
        edges_in_band.astype(np.float32), (feather_k, feather_k), 0,
    ) / 255.0
    soft_alpha = np.clip(soft_alpha, 0.0, 1.0)[..., None]

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    out = (img.astype(np.float32) * (1.0 - soft_alpha)
           + blurred.astype(np.float32) * soft_alpha)
    return out.astype(np.uint8)


def soften_object_outline_gc(img: np.ndarray, boxes_xyxy: np.ndarray,
                             iter_count: int = 5,
                             sigma_px: float = 14.0,
                             blur_ksize: int = 41) -> np.ndarray:
    """Plan B: GrabCut segmentation → distance-transform gradient → blend.

    For each ground-truth box:
      1. cv2.grabCut returns a foreground/background mask given the box
         as the foreground hint. Uses RGB color GMMs and an iterative
         graph cut — robust on textured backgrounds.
      2. Compute the signed distance from each pixel to the silhouette
         (negative inside object, positive outside, zero at boundary).
      3. Apply a Gaussian falloff around the silhouette:
              alpha = exp(-d^2 / (2 sigma^2))
         → peaks at 1.0 right at the silhouette, smoothly fades to 0
         deep inside the object and far in the background.
      4. Alpha-blend the locally-blurred image with the original. The
         object body and far background stay untouched; the silhouette
         becomes a smooth gradient instead of a sharp matte cut.

    sigma_px controls the gradient width — pixels within ~2*sigma_px of
    the silhouette are noticeably softened.
    """
    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    full_alpha = np.zeros((h, w), dtype=np.float32)

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        rw, rh = x2 - x1, y2 - y1
        if rw < 4 or rh < 4:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            img_bgr, mask, (x1, y1, rw, rh),
            bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT,
        )
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0,
        ).astype(np.uint8)
        if fg.sum() == 0:
            continue

        # Signed distance to silhouette: negative inside, positive outside.
        dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(255 - fg, cv2.DIST_L2, 3)
        signed = dist_out - dist_in

        alpha = np.exp(-(signed * signed) / (2.0 * sigma_px * sigma_px))
        full_alpha = np.maximum(full_alpha, np.clip(alpha, 0.0, 1.0))

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    alpha_3 = full_alpha[..., None]
    out = (img.astype(np.float32) * (1.0 - alpha_3)
           + blurred.astype(np.float32) * alpha_3)
    return out.astype(np.uint8)


def add_fake_horizon(img: np.ndarray, rng: np.random.Generator,
                     thickness_range: tuple = (3, 7),
                     brightness_range: tuple = (-100, 100)) -> np.ndarray:
    """Add a thin horizontal line at a random y in the middle 60% of the image.

    A subtle brightness offset along one row mimics a horizon — the L-2
    semantic head learned "objects appear near the sky/ground boundary"
    from composite training data. A fake horizon misdirects that bias.
    """
    h, w = img.shape[:2]
    y = int(rng.integers(int(h * 0.2), int(h * 0.8)))
    thickness = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
    delta = int(rng.integers(brightness_range[0], brightness_range[1] + 1))
    out = img.astype(np.int16)
    out[y:y + thickness, :, :] = np.clip(out[y:y + thickness, :, :] + delta, 0, 255)
    return out.astype(np.uint8)


def apply(img: np.ndarray, strength: float = 0.5,
          seed: int | None = None) -> np.ndarray:
    """Compose all 3 transforms with RNG gating by strength.

    strength=0.0 → no perturbation
    strength=1.0 → all 3 transforms always applied
    in between → each transform applied with probability `strength`

    `seed` makes the perturbation deterministic (important for reproducible
    training and for unit tests).
    """
    if strength <= 0.0:
        return img.copy()
    rng = np.random.default_rng(seed)
    out = img.copy()
    if rng.random() < strength:
        out = add_fake_halo(out, rng)
    if rng.random() < strength:
        out = add_sky_noise(out, rng)
    if rng.random() < strength:
        out = add_fake_horizon(out, rng)
    return out
