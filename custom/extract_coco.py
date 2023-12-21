import os
import sys
from pathlib import Path

from pycocotools.coco import COCO
import requests


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_coco_subset(
        coco_json_dir: Path,
        classes: list,
        # output_dir: str
):
    # instantiate COCO specifying the annotations json path
    path = Path(coco_json_dir)
    coco = COCO(path)
    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=classes)
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    # for im in images:
    #     img_data = requests.get(im['coco_url']).content
    #     with open(output_dir + im['file_name'], 'wb') as handler:
    #         handler.write(img_data)

    return imgIds


if __name__ == "__main__":
    get_coco_subset(
        coco_json_dir=ROOT / 'data/json/instances_train2017.json',
        classes=['person']
        # output_dir='dataset/coco/image/train'
    )
