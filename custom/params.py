"""
参数模块
  本模块用于指定训练和绘图的参数
"""


from settings import *


train_params = {
    "train_script": str(PROJECT_ROOT / "train.py"),
    "weights": str(WEIGHTS / "weights"),
    "cfg": str(MODELS / "yolov5s.yaml"),
    "data": str(DATA / "coco128.yaml"),
    "epochs": 5,
    "batch_size": 16,
}

print_params = {
    "path": str(TRAIN_RESULT / "exp" / "results.csv"),
    "y_column": "     metrics/mAP_0.5",
    "title": "Training Performance",
    "xlabel": "epoch",
    "ylabel": "mAP_0.5",
}