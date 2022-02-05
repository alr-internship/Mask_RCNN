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

def main(args):
    # Root directory of the project
    ROOT_DIR = os.path.abspath(__file__ + "/../../../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "samples/ycb/logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # %% [markdown]
    # ## Configurations

    # %%
    class YCBConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "ycb_video_dataset"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = args.gpus
        IMAGES_PER_GPU = args.imgs_per_gpu

        # Number of classes (including background)
        NUM_CLASSES = 1 + 21  # background + 3 shapes

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 480
        IMAGE_MAX_DIM = 640

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 32

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 500

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 5

        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        # USE_MINI_MASK = False
        
    config = YCBConfig()
    config.display()

    # %% [markdown]
    # ## Notebook Preferences

    # %%
    # %% [markdown]
    # ## Dataset
    # 
    # Create a synthetic dataset
    # 
    # Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
    # 
    # * load_image()
    # * load_mask()
    # * image_reference()

    # %%
    class YCBDataset(utils.Dataset):
        """Generates the shapes synthetic dataset. The dataset consists of simple
        shapes (triangles, squares, circles) placed randomly on a blank surface.
        The images are generated on the fly. No file access required.
        """

        def load_data(self, annotation_json, images_dir):
            """ Load the coco-like dataset from json
            Args:
                annotation_json: The path to the coco annotations json file
                images_dir: The directory holding the images referred to by the json file
            """
            # Load json from file
            json_file = open(annotation_json)
            coco_json = json.load(json_file)
            json_file.close()
            
            # Add the class names using the base method from utils.Dataset
            source_name = "ycb"
            for category in coco_json['categories']:
                class_id = category['id']
                class_name = category['name']
                if class_id < 1:
                    print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                    return
                
                self.add_class(source_name, class_id, class_name)
            
            # Get all annotations
            annotations = {}
            for annotation in coco_json['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(annotation)
            
            # Get all images and add them to the dataset
            seen_images = {}
            for image in coco_json['images']:
                image_id = image['id']
                if image_id in seen_images:
                    print("Warning: Skipping duplicate image id: {}".format(image))
                else:
                    seen_images[image_id] = image
                    try:
                        image_file_name = image['file_name']
                        image_width = image['width']
                        image_height = image['height']
                    except KeyError as key:
                        print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                    
                    image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                    image_annotations = annotations[image_id]
                    
                    # Add the image using the base method from utils.Dataset
                    self.add_image(
                        source=source_name,
                        image_id=image_id,
                        path=image_path,
                        width=image_width,
                        height=image_height,
                        annotations=image_annotations
                    )

        def load_mask(self, image_id):
            """ Load instance masks for the given image.
            MaskRCNN expects masks in the form of a bitmap [height, width, instances].
            Args:
                image_id: The id of the image to load masks for
            Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
            """
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']
            instance_masks = []
            class_ids = []
            
            for annotation in annotations:
                class_id = annotation['category_id']
                mask = Image.new('1', (image_info['width'], image_info['height']))
                mask_draw = ImageDraw.ImageDraw(mask, '1')
                for segmentation in annotation['segmentation']:
                    mask_draw.polygon(segmentation, fill=1)
                    bool_array = np.array(mask) > 0
                    instance_masks.append(bool_array)
                    class_ids.append(class_id)

            mask = np.dstack(instance_masks)
            class_ids = np.array(class_ids, dtype=np.int32)
            
            return mask, class_ids

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