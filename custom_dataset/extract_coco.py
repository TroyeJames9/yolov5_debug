import fiftyone.zoo as foz

# To download the COCO dataset for only the "person" and "car" classes
dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    classes=["person"],
    max_samples=50
)