import os
import sys
import json
import requests
import shutil
import argparse
import fiftyone.zoo as foz
from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import print_args


def findProjectRoot(current_dir=None, marker_file=".gitignore"):
    """
    Find the project root directory based on the presence of a marker file.

    Args:
        current_dir(str): The starting directory for the search.
        If not provided,the current working directory is used.
        marker_file(str): The name of the marker file that indicates the project root.

    Returns:
        current_dir(str): The absolute path to the project root directory, or None if not found.
    """
    if current_dir is None:
        current_dir = os.getcwd()

    while True:
        marker_path = os.path.join(current_dir, marker_file)
        if os.path.isfile(marker_path):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    return None


def setRelativePath(relative_path):
    """
    Set the relative path based on the project root directory.

    Args:
        relative_path(str): The relative path to set.

    Returns:
        absolute_path(str): The absolute path corresponding to the relative path.
        If there is no project root, an error will be raised.
    """
    project_root = findProjectRoot()

    if project_root is not None:
        absolute_path = os.path.join(project_root, relative_path)
        return absolute_path
    else:
        raise RuntimeError("Project root not found.")


def checkAndCreatePath(abs_path):
    """
    Check if abs_path exists, if not, create abs path recursively

    Args:
        abs_path: An absolute path

    Returns:
        None
    """
    # 判断路径是否存在
    if not os.path.exists(abs_path):
        # 递归创建路径
        os.makedirs(abs_path)
        print(f"Path '{abs_path}' created.")
    else:
        print(f"Path '{abs_path}' already exists.")
    return abs_path


def hasFilesInPath(path):
    """
    Check if path is empty

    Args:
        path: An absolute path

    Returns:
        Returns True if it is not empty, otherwise returns False
    """
    # 使用 os.scandir 检查路径是否包含文件
    return any(entry.is_file() for entry in os.scandir(path))


def getCocoSubsetV1(
        coco_json_dir: str,
        coco_json_name: str,
        classes: list,
        output_dir: str
):
    """
    Extract the coco subset containing specified classes based on coco json,
    and store the subset in output_dir

    Args:
        coco_json_dir: The parent directory(relative) of the coco json file
        coco_json_name: The coco json file name
        classes: the subset of the 80 object categories
        output_dir: Relative path to store subset

    Returns:
        None
    """
    # instantiate COCO specifying the annotations json path
    json_relative_path = coco_json_dir + '/' + coco_json_name
    json_path = setRelativePath(json_relative_path)
    output_path = setRelativePath(output_dir)
    checkAndCreatePath(output_path)
    coco = COCO(json_path)
    catIds = coco.getCatIds(catNms=classes)
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    if not hasFilesInPath(output_path):
        for im in images:
            img_data = requests.get(im['coco_url']).content
            output_file_path = os.path.join(output_path, im['file_name'])
            with open(output_file_path, 'wb') as handler:
                handler.write(img_data)


def getCocoSubsetV2(
        coco_name,
        dataset_dir,
        splits: list = None,
        classes: list = None,
        max_samples: int = None,
        only_matching=True
):
    """
    Extract the coco subset containing specified classes by fiftyone,
    and store the subset and json in dataset_dir.

    Args:
        coco_name: coco-2014 or coco-2017
        dataset_dir: Relative path to store subset and json
        splits: specifying the splits to load.Supported values are ("train", "test", "validation").If neither is provided, all available splits are loaded
        classes: specifying required classes to load.Default is person
        max_samples: a maximum number of samples to load per split
        only_matching: whether to only load labels that match the classes or attrs requirements that you provide

    Returns:
        None
    """
    dataset_path = setRelativePath(dataset_dir)
    checkAndCreatePath(dataset_path)
    foz.load_zoo_dataset(
        coco_name,
        splits=splits,
        classes=classes,
        only_matching=only_matching,
        max_samples=max_samples,
        dataset_dir=dataset_path
    )


def get_category_id(categories_name, json_data):
    """
    Return the id corresponding to categories_name based on json_data
    Args:
        categories_name(list): The subset of the 80 object categories
        json_data(json): Dictionary in json format

    Returns:
        list. The ids corresponding to categories_name,If the id cannot be found, raise error
    """
    category_id = []
    for cname in categories_name:
        for category in json_data["categories"]:
            if category["name"] == cname:
                category_id.append(category["id"])
                break
        else:
            raise ValueError(f"Category '{cname}' not found")
    return category_id


def filterAnnotation(json_file_path, classes):
    """
    In order to improve the speed of cocojson2yolovtxt, filter out the unnecessary parts of coco json in advance

    Args:
        json_file_path: Absolute path to cocojson file
        classes: the subset of the 80 object categories

    Returns:
        None

    """
    json_file_dir = os.path.dirname(json_file_path)
    with open(json_file_path, "r") as json_file:
        # Load the JSON data from the file
        json_data = json.load(json_file)
    if json_data.get('annotations') is not None:
        filtered_cat_id = get_category_id(classes, json_data)
        filtered_annotations = [
            {key: value for key, value in annotation.items() if key != "segmentation"}
            for annotation in json_data["annotations"]
            if annotation["category_id"] in filtered_cat_id
        ]
        json_data["annotations"] = filtered_annotations

    with open(json_file_dir + "_labels.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print("Filtered annotations saved")


def convertBboxCoco2Yolo(img_width: int, img_height: int, bbox: list):
    """
    Convert bounding box from COCO  format to YOLO format

    Args:
        img_width(int) : width of image
        img_height(int) : height of image
        bbox(list[int]) : bounding box annotation in COCO format:[top left x position, top left y position, width, height]

    Returns:
        list[float].bounding box annotation in YOLO format:[x_center_rel, y_center_rel, width_rel, height_rel]
    """

    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]


def convertCocoJson2YoloTxt(output_path, json_file, is_darknet):
    """
    Batch generate txt in yolov format based on cocojson in image units
    Args:
        output_path: Relative path to store yolov txt
        json_file: Absolute path to cocojson file
        is_darknet: Whether to generate _darkent.labels

    Returns:
        None
    """
    with open(json_file) as f:
        json_data = json.load(f)

    if json_data.get('annotations') is not None:
        # write _darknet.labels, which holds names of all classes (one class per line)
        if is_darknet:
            label_file = os.path.join(output_path, "_darknet.labels")
            with open(label_file, "w") as f:
                for category in tqdm(json_data["categories"], desc="Categories"):
                    category_name = category["name"]
                    f.write(f"{category_name}\n")

        for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
            img_id = image["id"]
            img_name = image["file_name"]
            img_width = image["width"]
            img_height = image["height"]

            anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
            anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
            with open(anno_txt, "w") as f:
                for anno in anno_in_image:
                    category = anno["category_id"]
                    bbox_COCO = anno["bbox"]
                    x, y, w, h = convertBboxCoco2Yolo(img_width, img_height, bbox_COCO)
                    f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print("Converting COCO Json to YOLO txt finished!")


def stdDataset(splits, classes, dataset_dir, is_darknet):
    """
    Standardize the file structure of coco subset and convert cocojson to yolov txt

    Args:
        splits: specifying the splits to load.Supported values are ("train", "test", "validation").
        dataset_dir: Relative path to store coco subset
        classes: the subset of the 80 object categories
        is_darknet: Whether to generate _darknet.labels
    Returns:
        None
    """
    dataset_path = setRelativePath(dataset_dir)
    images_path = {}
    labels_path = {}
    json_path = dataset_path + '/labels'
    for split in splits:
        images_path[split] = checkAndCreatePath(dataset_path + '/images/' + split)
        labels_path[split] = checkAndCreatePath(dataset_path + '/labels/' + split)
        if not os.path.exists(images_path[split] + '/data'):
            shutil.move(dataset_path + '/' + split + '/data/', images_path[split])
            shutil.move(dataset_path + '/' + split + '/labels.json', labels_path[split])
            shutil.rmtree(dataset_path + '/' + split)
        filterAnnotation(labels_path[split] + '/labels.json', classes=classes)
        convertCocoJson2YoloTxt(labels_path[split], json_path + '/' + split + '_labels.json', is_darknet=is_darknet)
        os.remove(labels_path[split] + '/labels.json')


def run(
        coco_name='coco-2017',
        dataset_dir='dataset/coco-2017',
        splits=None,
        classes='person',
        max_samples=None,
        is_darknet=False
):
    if splits is None:
        splits = ['train', 'validation']
    getCocoSubsetV2(
        coco_name,
        dataset_dir=dataset_dir,
        splits=splits,
        classes=classes,
        max_samples=max_samples
    )
    stdDataset(splits=splits, classes=classes, dataset_dir=dataset_dir, is_darknet=is_darknet)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-name', type=str, default='coco-2017',
                        help='The value is coco-2014 or coco-2017')
    parser.add_argument('--dataset-dir', type=str, default='dataset/coco-2017',
                        help='Relative path to store coco subset result')
    parser.add_argument('--splits', nargs='+', type=str, default=None, help='the splits to load.')
    parser.add_argument('--classes', nargs='+', type=str, default='person', help='classes to load.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='samples to load per split.')
    parser.add_argument('--is-darknet', action='store_true', help='Whether to generate _darkent.labels')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
