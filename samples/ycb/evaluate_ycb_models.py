import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
import numpy as np
from pathlib import Path


# Root directory of the project
ROOT_DIR = os.path.abspath(__file__ + "/../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from samples.ycb.train_ycb import YCBConfig
from samples.ycb.ycb_dataset import YCBDataset


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = YCBConfig(gpus=1, imgs_per_gpu=1)
config.display()

# Training dataset
data_dir = Path("data/YCB_Video_Dataset")

# Validation dataset
dataset_val = YCBDataset()
dataset_val.load_data(data_dir / "annotations/val_instances.json", data_dir / "data")
dataset_val.prepare()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
dir_path = Path(ROOT_DIR) / "resources/ycb/ycb_video_dataset20220205T1517"
model_checkpoints = dir_path.glob("*.h5")

for model_checkpoint in model_checkpoints:
    # Load trained weights
    print("Loading weights from ", model_checkpoint)
    model.load_weights(model_checkpoint, by_name=True)

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 100)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config,
                                image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print(f"Checkpoint {model_checkpoint.name}, mAP: {np.mean(APs)}")