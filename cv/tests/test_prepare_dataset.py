from prepare_dataset import coco_to_yolo_line, make_splits


def test_coco_to_yolo_line_normalizes_and_centers():
    ann = {"category_id": 3, "bbox": [1349.0, 162.0, 173.0, 42.0]}
    cat_map = {i: i for i in range(18)}
    line = coco_to_yolo_line(ann, img_w=1920, img_h=1080, cat_map=cat_map)
    assert line == "3 0.747656 0.169444 0.090104 0.038889"


def test_coco_to_yolo_line_remaps_category_id():
    ann = {"category_id": 50, "bbox": [0.0, 0.0, 192.0, 108.0]}
    cat_map = {50: 7}
    line = coco_to_yolo_line(ann, img_w=1920, img_h=1080, cat_map=cat_map)
    assert line.startswith("7 ")


def test_make_splits_is_deterministic_and_partitions():
    ids = list(range(100))
    a = make_splits(ids, val_frac=0.2, seed=42)
    b = make_splits(ids, val_frac=0.2, seed=42)
    assert a == b
    assert len(a["val"]) == 20 and len(a["train"]) == 80
    assert set(a["val"]) | set(a["train"]) == set(ids)
    assert not (set(a["val"]) & set(a["train"]))
