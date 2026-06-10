"""Bounding-box format conversions for the CV pipeline."""


def xyxy_to_ltwh(x1, y1, x2, y2, img_w, img_h):
    """Convert an xyxy box to LTWH pixel coords, clamped to image bounds.

    Returns [left, top, width, height] as floats.
    """
    x1 = min(max(float(x1), 0.0), float(img_w))
    y1 = min(max(float(y1), 0.0), float(img_h))
    x2 = min(max(float(x2), 0.0), float(img_w))
    y2 = min(max(float(y2), 0.0), float(img_h))
    return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def coco_to_yolo(coco_box, img_w, img_h):
    """Convert a COCO [x, y, w, h] absolute box to YOLO normalized
    (cx, cy, w, h). Returns a 4-tuple of floats."""
    x, y, w, h = coco_box
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return (cx, cy, w / img_w, h / img_h)
