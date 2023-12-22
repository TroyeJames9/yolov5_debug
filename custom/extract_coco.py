import os
import re
from pycocotools.coco import COCO
import fiftyone.zoo as foz
import fiftyone as fo
import requests


def find_project_root(current_dir=None, marker_file=".gitignore"):
    """
    Find the project root directory based on the presence of a marker file.

    :param current_dir: The starting directory for the search. If not provided, the current working directory is used.
    :param marker_file: The name of the marker file that indicates the project root.
    :return: The absolute path to the project root directory, or None if not found.
    """
    if current_dir is None:
        current_dir = os.getcwd()

    while True:
        # Check if the marker file exists in the current directory
        marker_path = os.path.join(current_dir, marker_file)
        if os.path.isfile(marker_path):
            return current_dir

        # Move up one level in the directory hierarchy
        parent_dir = os.path.dirname(current_dir)

        # Break the loop if we have reached the root directory
        if parent_dir == current_dir:
            break

        current_dir = parent_dir

    return None


def set_relative_path(relative_path):
    """
    Set the relative path based on the project root directory.

    :param relative_path: The relative path to set.
    """
    # Find the project root
    project_root = find_project_root()

    if project_root is not None:
        # Combine the project root and the relative path
        absolute_path = os.path.join(project_root, relative_path)
        # absolute_path = normalize_path(absolute_path)
        return absolute_path
    else:
        raise RuntimeError("Project root not found.")


def normalize_path(path):
    # 将正斜杠替换为反斜杠
    path = path.replace('/', '\\')
    # 将连续两个反斜杠替换为一个反斜杠
    path = re.sub(r'\\\\', r'\\', path)
    return path


def check_and_create_path(abs_path):
    # 判断路径是否存在
    if not os.path.exists(abs_path):
        # 递归创建路径
        os.makedirs(abs_path)
        print(f"Path '{abs_path}' created.")
    else:
        print(f"Path '{abs_path}' already exists.")


def has_files_in_path(path):
    # 使用 os.scandir 检查路径是否包含文件
    return any(entry.is_file() for entry in os.scandir(path))


def get_coco_subset_v1(
        coco_json_dir: str,
        coco_json_name: str,
        classes: list,
        output_dir: str
):
    # instantiate COCO specifying the annotations json path
    json_relative_path = coco_json_dir + '/' + coco_json_name
    json_path = set_relative_path(json_relative_path)
    output_path = set_relative_path(output_dir)
    check_and_create_path(output_path)
    coco = COCO(json_path)

    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=classes)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    if not has_files_in_path(output_path):
        for im in images:
            img_data = requests.get(im['coco_url']).content
            output_file_path = os.path.join(output_path, im['file_name'])
            with open(output_file_path, 'wb') as handler:
                handler.write(img_data)

    return imgIds


def get_coco_subset_v2(
        coco_name='coco-2017',
        dataset_dir='dataset/coco-2017',
        splits=None,
        classes=None,
        max_samples=None,
        only_matching=True,

):
    dataset_path = set_relative_path(dataset_dir)
    check_and_create_path(dataset_path)
    dataset = foz.load_zoo_dataset(
        coco_name,
        splits=splits,
        classes=classes,
        only_matching=only_matching,
        max_samples=max_samples,
        dataset_dir=dataset_path
    )


if __name__ == "__main__":
    # imgids= get_coco_subset(
    #     coco_json_dir='data/json',
    #     coco_json_name='instances_train2017.json',
    #     classes=['person'],
    #     output_dir='dataset/coco/image/train'
    # )
    get_coco_subset_v2(classes='person', splits=['train', 'validation'])
