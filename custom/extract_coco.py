import os
import re
import json
import requests
import shutil
import fiftyone.zoo as foz
from pycocotools.coco import COCO
from tqdm import tqdm


def findProjectRoot(current_dir=None, marker_file=".gitignore"):
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


def setRelativePath(relative_path):
    """
    Set the relative path based on the project root directory.

    :param relative_path: The relative path to set.
    """
    # Find the project root
    project_root = findProjectRoot()

    if project_root is not None:
        # Combine the project root and the relative path
        absolute_path = os.path.join(project_root, relative_path)
        # absolute_path = normalize_path(absolute_path)
        return absolute_path
    else:
        raise RuntimeError("Project root not found.")


def normalizePath(path):
    # 将正斜杠替换为反斜杠
    path = path.replace('/', '\\')
    # 将连续两个反斜杠替换为一个反斜杠
    path = re.sub(r'\\\\', r'\\', path)
    return path


def checkAndCreatePath(abs_path):
    # 判断路径是否存在
    if not os.path.exists(abs_path):
        # 递归创建路径
        os.makedirs(abs_path)
        print(f"Path '{abs_path}' created.")
    else:
        print(f"Path '{abs_path}' already exists.")
    return abs_path


def hasFilesInPath(path):
    # 使用 os.scandir 检查路径是否包含文件
    return any(entry.is_file() for entry in os.scandir(path))


def getCocoSubsetV1(
        coco_json_dir: str,
        coco_json_name: str,
        classes: list,
        output_dir: str
):
    # instantiate COCO specifying the annotations json path
    json_relative_path = coco_json_dir + '/' + coco_json_name
    json_path = setRelativePath(json_relative_path)
    output_path = setRelativePath(output_dir)
    checkAndCreatePath(output_path)
    coco = COCO(json_path)

    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=classes)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    if not hasFilesInPath(output_path):
        for im in images:
            img_data = requests.get(im['coco_url']).content
            output_file_path = os.path.join(output_path, im['file_name'])
            with open(output_file_path, 'wb') as handler:
                handler.write(img_data)

    return imgIds


def getCocoSubsetV2(
        coco_name='coco-2017',
        dataset_dir='dataset/coco-2017',
        splits=None,
        classes=None,
        max_samples=None,
        only_matching=True
):
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


def convertBboxCoco2Yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format:
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format:
        [x_center_rel, y_center_rel, width_rel, height_rel]
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


def convertCocoJson2YoloTxt(output_path, json_file):
    with open(json_file) as f:
        json_data = json.load(f)

    # write _darknet.labels, which holds names of all classes (one class per line)
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


def filterAnnotation(json_file):
    filtered_annotations = [annotation for annotation in json_file["annotations"] if annotation["category_id"] == 1]
    json_file["annotations"] = filtered_annotations
    with open("filtered_labels.json", "w") as json_file:
        json.dump(json_file, json_file, indent=2)
    print("Filtered annotations saved to 'filtered_annotations.json'")


def stdDataset(
        dataset_dir='dataset/coco-2017'
):
    dataset_path = setRelativePath(dataset_dir)
    images_train_path = checkAndCreatePath(setRelativePath(dataset_dir + '/images/train'))
    images_val_path = checkAndCreatePath(setRelativePath(dataset_dir + '/images/val'))
    labels_train_path = checkAndCreatePath(setRelativePath(dataset_dir + '/labels/train'))
    labels_val_path = checkAndCreatePath(setRelativePath(dataset_dir + '/labels/val'))
    shutil.move(dataset_path + '/train/data/', images_train_path)
    shutil.move(dataset_path + '/validation/data/', images_val_path)
    shutil.move(dataset_path + '/train/labels.json', labels_train_path)
    shutil.move(dataset_path + '/validation/labels.json', labels_val_path)
    shutil.rmtree(dataset_path + '/train')
    shutil.rmtree(dataset_path + '/validation')
    convertCocoJson2YoloTxt(labels_train_path, labels_train_path + '/labels.json')
    convertCocoJson2YoloTxt(labels_val_path, labels_val_path + '/labels.json')


if __name__ == "__main__":
    # getCocoSubsetV2(splits=['train', 'validation'], classes='person')
    # stdDataset()
    pass
