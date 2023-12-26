# About The Project

This is the repo of the campus project - research on the application of dynamic people flow statistics in campus. 
This project uses the **[yolov5](https://github.com/ultralytics/yolov5)** to combine with deep-sort, adds an attention mechanism module to yolov5. In deep-sort
Basically, the Kalman filter is optimized, and Fast-ReID is finally introduced for tracking and matching to improve
the stability of target matching.

The main contributions of this repo are:
* Based on **[fiftyone](https://docs.voxel51.com/integrations/coco.html)** to download the COCO subset and 
create a one-click script to **generate the standard yolov format data set file structure**.
* etc...

<!-- GETTING STARTED -->
# Getting Started

This section guides you on how to use this project from dataset construction, 
model training to integration with WEB visualization pages.

## Prerequisites

Before starting this project, you need to:
* git clone 
  ```sh
  git clone https://github.com/TroyeJames9/yolov5_debug.git
  ```

* Install dependency packages in pycharm
  ```sh
  pip install -r requirements.txt
  ```
## Dataset construction(COCO subset)

This project requires using COCO's person subset to fine-tune the yolov 5 model.  
# _editing..._


