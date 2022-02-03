#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:42:15 2019

@author: Zhiyu Ye

Email: yezhiyu@hotmail.com

In London, the United Kingdom
"""

import argparse
import logging
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.sub_masks_annotations import create_sub_masks, create_sub_mask_annotation
import random
from joblib import Parallel, delayed, cpu_count


def process_to_file(height, width, image_dir, categories, iscrowd,
                    image_names, image_id_index, input_dir, filename, jobs):

    def process_image(image_name, image_id):
        logging.info(f'Processing: {image_name} ...')

        # Write infomation of each image
        file_name = image_name + '-color.png'
        image_item = {'file_name': file_name, 'height': height, 'id': image_id, 'width': width}

        # Write information of each mask in the image
        mask_name = image_name + '-label.png'
        image = Image.open(image_dir + '/' + mask_name)
        # Extract each mask of the image
        sub_masks = create_sub_masks(image)
        return sub_masks, image_item

    def process_annotations(image_names, image_ids):
        annotations = []
        images = []
        for image_id, image_name in zip(image_ids, image_names):
            sub_masks, image_item = process_image(image_name, image_id)
            images.append(image_item)
            annotation_id = image_id * len(categories)

            for category_id, sub_masks in sub_masks.items():
                category_id = int(category_id[1:category_id.find(',')])
                cimg = np.array(sub_masks)
                opencvImage = np.stack((cimg, cimg, cimg), axis=2)
                instance = np.uint8(np.where(opencvImage == True, 0, 255))
                annotation_item = create_sub_mask_annotation(
                    instance, image_id, category_id, annotation_id, iscrowd)
                annotations.append(annotation_item)
                annotation_id += 1
        return annotations, images

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    image_names_chuncked = list(split(image_names, jobs))
    image_ids_chuncked = list(split(image_id_index, jobs))

    return_value = Parallel(n_jobs=jobs)(
        delayed(process_annotations)(image_names_chunk, image_ids_chunk)
        for image_names_chunk, image_ids_chunk
        in zip(image_names_chuncked, image_ids_chuncked)
    )

    annotations, images = zip(*return_value)
    annotations = [annotation for annotations_l in annotations
                   for annotation in annotations_l]
    images = [image for images_l in images
              for image in images_l]

    # logging.info(f'Test if all the instances are detected, the result is {count == annotation_id}')
    # Combine categories, annotations and images to form a json file
    json_data = {'annotations': annotations, 'categories': categories, 'images': images}
    annotations_output_dir = input_dir + '/annotations'
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir)
    output_json_file = annotations_output_dir + f'/{filename}.json' 
    logging.info(f"saving json to {output_json_file}")
    with open(output_json_file, 'w') as f:
        json.dump(json_data, f)


def main(args):
    input_dir = (Path(__file__).parent / 'data/YCB_Video_Dataset').as_posix()

    # Generate the categories
    class_file = open(input_dir + '/image_sets/classes.txt')
    line = class_file.readline()
    category_id = 0
    categories = []
    while line:
        category_id += 1
        category = {'supercategory': line, 'id': category_id, 'name': line}
        categories.append(category)
        line = class_file.readline()
    class_file.close()

    # Read the names of the images to generator annotations
    image_names_file = open(input_dir + '/image_sets/train.txt')
    line = image_names_file.readline()
    image_names = []
    while line:
        image_names.append(line[:-1])
        line = image_names_file.readline()
    image_names_file.close()

    num_of_images = len(image_names)

    # generate indices
    image_id_index = list(range(num_of_images))

    # shuffle data
    random.shuffle(image_names)

    # Generate the images and the annotations
    image_dir = input_dir + '/data'
    width = 640
    height = 480
    iscrowd = 0

    params = dict(height=height, width=width, image_dir=image_dir, categories=categories,
                  iscrowd=iscrowd, input_dir=input_dir, jobs=args.jobs)

    # random split
    ds_border = num_of_images // 20
    # assert num_of_images == (len(val_indices) + len(train_indices))

    process_to_file(image_names=np.array(image_names)[:ds_border],
                    image_id_index=np.array(image_id_index)[:ds_border].tolist(),
                    filename="val_instances", **params)

    process_to_file(image_names=np.array(image_names)[ds_border:],
                    image_id_index=np.array(image_id_index)[ds_border:].tolist(),
                    filename="train_instances", **params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = argparse.ArgumentParser()
    args.add_argument("--jobs", type=int, default=cpu_count())
    main(args.parse_args())
