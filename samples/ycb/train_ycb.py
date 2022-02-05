# %% [markdown]
# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# %%
from argparse import ArgumentParser
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
import time
import numpy as np
import json
import wandb
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath(__file__ + "/../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from samples.ycb.ycb_config import YCBConfig
from samples.ycb.ycb_dataset import YCBDataset
import mrcnn.model as modellib


def main(args):

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "samples/ycb/logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "resources/coco/mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # %% [markdown]
    # ## Configurations

    # %%
        
    config = YCBConfig()
    config.display()

    # wandb to monitor system utilization
    experiment = wandb.init(project="MaskRCNN", resume='allow', entity="depth-denoising", reinit=True)

    # %%
    # Training dataset
    data_dir = Path(ROOT_DIR + "/samples/ycb/data/YCB_Video_Dataset")
    train_file = data_dir / "annotations/train_instances.json"
    val_file = data_dir / "annotations/val_instances.json"

    print(f"Loading train dataset: {train_file}")
    dataset_train = YCBDataset()
    dataset_train.load_data(train_file, data_dir / "data")
    dataset_train.prepare()

    # Validation dataset
    print(f"Loading validation dataset: {val_file}")
    dataset_val = YCBDataset()
    dataset_val.load_data(val_file, data_dir / "data")
    dataset_val.prepare()

    # %% [markdown]
    # ## Create Model

    # %%
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)

    # %%
    # Which weights to start with?
    # init_with = "coco"  # imagenet, coco, or last
    init_with = "resources/ycb/mask_rcnn_ycb_video_dataset_0008.h5"

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    else:
        model_path = os.path.join(ROOT_DIR, init_with)
        model.load_weights(model_path, by_name=True)

    # %% [markdown]
    # ## Training
    # 
    # Train in two stages:
    # 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
    # 
    # 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

    # %%
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    start_train = time.time()
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=20, 
                layers='heads')
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')

    # %%
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    start_train = time.time()
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60,
                layers="all")
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')

    # %%
    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--gpus", type=int, default=1)
    argparse.add_argument("--imgs-per-gpu", type=int, default=8)
    main(argparse.parse_args())