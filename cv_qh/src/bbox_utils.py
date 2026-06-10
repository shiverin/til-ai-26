"""Bounding-box format conversions for the CV pipeline."""


def xyxy_to_ltwh_int(x1, y1, x2, y2, img_w, img_h):
    """Convert an xyxy box to integer LTWH pixel coords, clipped to image bounds.

    Returns [left, top, width, height] as ints. width/height are 0 when the
    clipped box degenerates — caller should filter those out.
    """
    x1 = min(max(float(x1), 0.0), float(img_w))
    y1 = min(max(float(y1), 0.0), float(img_h))
    x2 = min(max(float(x2), 0.0), float(img_w))
    y2 = min(max(float(y2), 0.0), float(img_h))
    left, top = int(round(x1)), int(round(y1))
    width = int(round(x2 - x1))
    height = int(round(y2 - y1))
    if left + width > img_w:
        width = img_w - left
    if top + height > img_h:
        height = img_h - top
    return [left, top, max(width, 0), max(height, 0)]
