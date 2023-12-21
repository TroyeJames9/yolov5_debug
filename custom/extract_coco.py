import os
import sys
from pathlib import Path
from pycocotools.coco import COCO
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
        return absolute_path
    else:
        raise RuntimeError("Project root not found.")


def get_coco_subset(
        coco_json_dir: str,
        coco_json_name: str,
        classes: list,
        # output_dir: str
):
    # instantiate COCO specifying the annotations json path
    json_relative_path = coco_json_dir + '/' + coco_json_name
    json_path = set_relative_path(json_relative_path)
    coco = COCO(json_path)
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
        coco_json_dir='data/json',
        coco_json_name='instances_train2017.json',
        classes=['person']
        # output_dir='dataset/coco/image/train'
    )
