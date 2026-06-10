from sweep_conf import build_coco_results, parse_confs, pick_best_conf


def test_build_coco_results_forces_score_one():
    dets = [(5, 3, [1.0, 2.0, 3.0, 4.0]), (5, 7, [0.0, 0.0, 9.0, 9.0])]
    results = build_coco_results(dets)
    assert all(r["score"] == 1.0 for r in results)
    assert results[0] == {
        "image_id": 5, "category_id": 3,
        "bbox": [1.0, 2.0, 3.0, 4.0], "score": 1.0,
    }


def test_pick_best_conf_returns_argmax():
    assert pick_best_conf({0.1: 0.50, 0.2: 0.71, 0.3: 0.64}) == 0.2


def test_parse_confs_sorts_and_deduplicates():
    assert parse_confs("0.45, 0.30,0.45") == [0.3, 0.45]
