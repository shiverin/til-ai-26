"""Class-balanced sampling for ultralytics YOLO training.

Defines YOLOWeightedDataset — a YOLODataset that samples training images
inversely to class frequency — plus a monkeypatch to activate it.

Template: https://y-t-g.github.io/tutorials/yolo-class-balancing/
"""
import numpy as np
from ultralytics.data.dataset import YOLODataset


def compute_class_weights(counts):
    """Weight each class inversely to its instance count (total / count)."""
    counts = np.asarray(counts, dtype=float)
    counts = np.where(counts == 0, 1.0, counts)  # avoid divide-by-zero
    return counts.sum() / counts


def compute_image_weights(image_class_indices, class_weights, agg=np.mean):
    """One weight per image, aggregating the class weights of its labels.
    `image_class_indices` is a per-image list of class-index sequences;
    empty images get a neutral weight of 1.0."""
    weights = []
    for classes in image_class_indices:
        classes = np.asarray(classes, dtype=int)
        if classes.size == 0:
            weights.append(1.0)
        else:
            weights.append(float(agg(class_weights[classes])))
    return np.asarray(weights, dtype=float)


def to_probabilities(weights):
    """Normalize a weight array into a probability distribution."""
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


class YOLOWeightedDataset(YOLODataset):
    """YOLODataset that draws *training* images by inverse-class-frequency
    probability. Validation indexing stays sequential (unweighted).

    Caveat: mosaic pulls its other tiles via unweighted random indices, so
    balancing is partial — see SOLUTION.md §8."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.agg_func = np.mean  # toggle to np.sum to balance more aggressively
        class_indices = [
            lbl["cls"].reshape(-1).astype(int) for lbl in self.labels
        ]
        n_classes = len(self.data["names"])
        flat = (
            np.concatenate(class_indices)
            if class_indices else np.array([], dtype=int)
        )
        counts = np.bincount(flat, minlength=n_classes)
        self.class_weights = compute_class_weights(counts)
        self.image_weights = compute_image_weights(
            class_indices, self.class_weights, self.agg_func
        )
        self.probabilities = to_probabilities(self.image_weights)

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(index))


def patch_weighted_dataset():
    """Monkeypatch ultralytics to use YOLOWeightedDataset for training."""
    import ultralytics.data.build as build
    build.YOLODataset = YOLOWeightedDataset
    print("[finetune] weighted (class-balanced) dataloader active")
