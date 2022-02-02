Train MaskRCNN on the YCB Video Dataset
=====

Setup
----

1. Install conda environment
    ```bash
    conda env create -f environment.yml
    ```

    The conda environment is the easiest way to get all packages with the required version.
    Nevertheless, one can install major packages on their own.
    The most important packages with the correct version are the following
    - python 3.6
    - tensorflow 1.5.0
    - keras 2.1.5
    - scikit-image 0.16.2
    - h5py 2.10.0
    Make sure to install exactly these versions, as different versions will not work in many cases.

1. Acivate conda environment

    conda activate mask_rcnn

Training

1. Download YCB Video Dataset

    The YCB Video Dataset can be downloaded from [https://rse-lab.cs.washington.edu/projects/posecnn/](https://rse-lab.cs.washington.edu/projects/posecnn/).
    The whole dataset is stored on google drive as ZIP file and has approximatly 265 GB of size.

    Unzip the downloaded dataset and copy or symlink it to `samples/ycb/data/YCB_Video_Dataset`

1. Generate Coco-Like Annotations

    To train the MaskRCNN, it is the easiest to convert the YCB Video Dataset annotations
    to the format that was also used in the COCO dataset.
    Therefore execute the [video_data_annotations_generator.py](samples/ycb/video_data_annotations_generator.py).
    This script can take more than an hour.
    Set the number of threads to use with `--jobs=%NUM_JOBS%`.

1. Train MaskRCNN

    The MaskRCNN can be trained by executing
    ```commandline
        python samples/ycb/train_ycb.py
    ```