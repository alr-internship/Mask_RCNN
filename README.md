Mask R-CNN trained on YCB Video Dataset
====

Detection example (trained for 8 epochs):
![](assets/img1.png)

Setup
----

1. Install Packages
    1. conda environment
        There are two conda environments.
        One for inference on gpu, one for inference on cpu.
        The different environments are needed, 
        as there are separate tensorflow packages for cpu/gpu.

        - For CPU inference install the `mask_rcnn_cpu` environment:
            ```bash
            conda env create -f environment_cpu.yml
            ```
        - For GPU inference install the `mask_rcnn_gpu` environment:
            ```bash
            conda env create -f environment_gpu.yml
            ```
    
    2. Manual installation
        The conda environments are the easiest way to get all packages with the required versions.
        Nevertheless, one can install major packages on their own.
        The most important packages with the correct versions are the following:

        - python 3.6
        - tensorflow 1.15.0 or tensorflow-gpu 1.15.0
        - keras 2.3.1
        - scikit-image 0.16.2
        - h5py 2.10.0

        Make sure to install exactly these versions, as different versions will not work in many cases.
        Lookup the [environment_cpu.yml](environment_cpu.yml) or [environment_gpu.yml](environment_gpu.yml)
        for further, required packages that need to be installed.

1. Activate conda environment

    - For CPU inference
        ```bash
        conda activate mask_rcnn_cpu
        ```
    - For GPU inference
        ```bash
        conda activate mask_rcnn_gpu
        ```

Training
---

1. Download YCB Video Dataset

    The YCB Video Dataset can be downloaded from [https://rse-lab.cs.washington.edu/projects/posecnn/](https://rse-lab.cs.washington.edu/projects/posecnn/).
    The whole dataset is stored on google drive as ZIP file and has approximatly 265 GB of size.

    Unzip the downloaded dataset and copy or symlink it to `samples/ycb/data/YCB_Video_Dataset`

1. Generate Coco-Like Annotations

    Activate the conda environment.
    To train the MaskRCNN, it is the easiest to convert the YCB Video Dataset annotations
    to the format that was also used in the COCO dataset.
    Therefore execute the [video_data_annotations_generator.py](samples/ycb/video_data_annotations_generator.py).
    This script can take more than an hour to generate the required .json files for the validation and train datasets.
    Set the number of threads to use with `--jobs=%NUM_JOBS%`.
    Per default, the script splits the YCB Video Dataset into the validation and train dataset,
    where the validation dataset consists of 5 % and the train dataset of 95 % of the samples.

1. Train MaskRCNN

    Activate the conda environment.
    The MaskRCNN can be trained by executing
    ```commandline
        export KERAS_BACKEND=tensorflow
        python samples/ycb/train_ycb.py
    ```
    Per default, a snapshot of the model will be made every epoch.
    They can be found in the [logs](samples/ycb/logs) folder.

Test
----

To test the trained MaskRCNN, one can execute the [test_ycb.ipynb](samples/ycb/test_ycb.ipynb).
Make sure to adapt the path to the generate snapshot of the model (`.h5` extension).
Per default, samples from the validation dataset will be used to test the model.

Evaluate
----

The evaluation of the trained model snapshots can be conducted with the [evaluate_ycb_models.py](samples/ycb/evaluate_ycb_models.py). This file evaluates a directory containing all the snapshots. It will compute the `mAP` on 100 randomly chosen samples from the validation dataset for every snapshot.
The evaluation results will be printed to the stdout

BwUniCluster 2.0
----

The majority of scripts can be executed in the [BwUniCluster 2.0](https://wiki.bwhpc.de/e/Main_Page).
All relevant scripts are located in [bwunicluster](bwunicluster)

