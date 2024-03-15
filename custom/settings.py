"""所有跨文件的全局变量由本模块计算并赋值

其他模块需要使用本模块全局变量时，在模块开头导入本模块即可
例子：
    from setting import *...

"""

from pathlib import Path

# 获取当前文件的绝对路径
ROOT = Path(__file__).resolve()

# 定义项目的根目录
PROJECT_ROOT = ROOT.parent.parent
print(PROJECT_ROOT)
# 定义项目中的各个目录和文件名
WEIGHTS = PROJECT_ROOT / "models" / "weights"
DATA = PROJECT_ROOT / "data"
MODELS = PROJECT_ROOT / "models"
DATASETS = PROJECT_ROOT / "datasets"
TRAIN_RESULT = PROJECT_ROOT / "runs" / "train"


