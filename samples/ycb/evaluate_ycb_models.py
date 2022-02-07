from argparse import ArgumentParser
import csv
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


def main(args):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    config = YCBConfig(gpus=1, imgs_per_gpu=1)
    config.display()

    # Training dataset
    data_dir = Path(ROOT_DIR + "/samples/ycb/data/YCB_Video_Dataset")

    # Validation dataset
    dataset_test = YCBDataset()
    dataset_test.load_data(data_dir / "annotations/test_instances.json", data_dir / "data")
    dataset_test.prepare()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=config,
                            model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_checkpoints = sorted(args.models_dir.glob("*.h5"))

    metrics = []
    for model_checkpoint in model_checkpoints:
        # Load trained weights
        print("Loading weights from ", model_checkpoint)
        model.load_weights(model_checkpoint.as_posix(), by_name=True)

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on N images. Increase for better accuracy.
        APs = []
        for image_id in dataset_test.image_ids:
            # Load image and ground truth data
            image, _, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, config,
                                    image_id, use_mini_mask=False)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, _, _, _ =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])

            APs.append(AP)
        metrics.append({
            "model": model_checkpoint.stem,
            "mAP": np.mean(APs),
            "sAP": np.std(APs)
        })

    with open(f'{args.models_dir}/eval.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("models_dir", type=Path)
    main(argparse.parse_args())