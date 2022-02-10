import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath(__file__ + "/../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config

class YCBConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ycb_video_dataset"


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


    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    # USE_MINI_MASK = False

    def __init__(self, gpus: int, imgs_per_gpu: int):
        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        self.GPU_COUNT = gpus
        self.IMAGES_PER_GPU = imgs_per_gpu
        # Use a small epoch since the data is simple
        self.STEPS_PER_EPOCH = 500 // self.GPU_COUNT
        super().__init__()
