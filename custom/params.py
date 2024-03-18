"""
参数模块
  本模块用于指定训练和绘图的参数
"""


from settings import *

# 训练参数
train_params = {
    "train_script": str(PROJECT_ROOT / "train.py"),
    "weights": str(WEIGHTS / "weights"),  # str: 预训练权重文件路径
    "cfg": str(MODELS / "yolov5s.yaml"),  # str: 模型配置文件路径
    "data": str(DATA / "coco128.yaml"),  # str: 数据集路径
    "epochs": 5,  # int: 训练轮数
    "batch_size": 16,  # int: 批次大小
}

# 绘图参数
print_params = {
    "path": str(TRAIN_RESULT / "exp" / "results.csv"),  # 确认run/train文件夹内的exp最大大小，并在其大小上加一，如最大exp4为，则次参数为exp5
    "y_column": "     metrics/mAP_0.5",  # str: 模型配置文件路径
    "title": "Training Performance",  # str: y轴的值
    "xlabel": "epoch",  # str: x轴的名字
    "ylabel": "mAP_0.5",  # str: y轴的名字
}