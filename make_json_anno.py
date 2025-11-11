'''
This file is for creating [MSCOCO style] json annotation files for convenient
training in popular detection frameworks, such as mmdetection, detectron,
maskrcnn-benchmark, etc. Before running this file, please follow the instructions
in https://github.com/waspinator/coco to install the COCO API or use the following
commands for installation:
    pip install cython
    pip install git+git://github.com/waspinator/coco.git@2.1.0

Usage: python3 code/make_json_anno.py --list_path /path/to/img/list/ [--tb_only]
'''

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from argparse import ArgumentParser
import pycococreatortools


'''
ActiveTuberculosis: Active TB
ObsoletePulmonaryTuberculosis: Latent TB
PulmonaryTuberculosis: Unknown TB
'''
def cat2label(cls_name):
    x = {'ActiveTuberculosis': 1, 'ObsoletePulmonaryTuberculosis': 2, 'PulmonaryTuberculosis': 3}
    return x[cls_name]


'''
Load annotations in the XML format
Input:
       xml_path: (string), xml annoation (relative) path
       size    : (int, int), align with the actual image size
'''
def load_annotation(xml_path, resized=(512, 512)):
    if not os.path.exists(xml_path):
        return None, None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = cat2label(name)
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        width_ratio = width / resized[1]
        height_ratio = height / resized[0]
        ignore = False
        bbox[0] /= width_ratio; bbox[2] /= width_ratio
        bbox[1] /= height_ratio; bbox[3] /= height_ratio
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 1 or h < 1:
            ignore = 1
        if difficult or ignore:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = None #np.zeros((0, 4))
        labels = None #np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    if bboxes is not None:
        return bboxes.astype(np.float32), labels.astype(np.int64)
    else:
        return None, None


def dataset_info():
    INFO = [
        {
            'contributor': 'Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng',
            'date_created': '2020/06/22',
            'description': 'TBX11K Dataset',
            'url': 'http://mmcheng.net/tb',
            'version': '1.0',
            'year': 2020
        }
    ]


    LICENSES = [
        {
            'id': 1,
            'name': 'Attribution-NonCommercial-ShareAlike License',
            'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'
        }
    ]

    CATEGORIES = [
        {'id': 1, 'name': 'ActiveTuberculosis', 'supercategory': 'Tuberculosis'},
        {'id': 2, 'name': 'ObsoletePulmonaryTuberculosis', 'supercategory': 'Tuberculosis'},
        {'id': 3, 'name': 'PulmonaryTuberculosis', 'supercategory': 'Tuberculosis'},
    ]

    return INFO, LICENSES, CATEGORIES


def main():
    parser = ArgumentParser()
    parser.add_argument('--list_path', default='lists/TBX11K_train.txt', help='image list file name')
    parser.add_argument('--anno_dir', default='annotations/xml/', help='annotation directory')
    parser.add_argument('--anno_savedir', default='annotations/json/', help='annotation directory')
    parser.add_argument('--tb_only', action='store_true', help='only for tuberculosis data')
    args = parser.parse_args()
    print('Processing {} with the tb_only option {}\n'.format(args.list_path, 'ON' if args.tb_only else 'OFF'))
    info, lic, classes = dataset_info()
    if not os.path.exists(args.anno_savedir):
        os.makedirs(args.anno_savedir)
    train_o = {
                'info': info,
                'licenses': lic,
                'categories': classes,
                'images': [],
                'annotations': []
    }
    resized_width = 512
    resized_height = 512
    files = open(args.list_path, 'r').read().splitlines()
    train_cnt = 1
    train_instance_cnt = 1
    for idx, file in enumerate(tqdm(files)):
        h, w = (resized_height, resized_width)
        file_name = os.path.basename(file)
        image_info = pycococreatortools.create_image_info(
                train_cnt, file, (int(w), int(h)))
        boxes, labels = load_annotation(args.anno_dir + file_name[:-3] + 'xml', resized=(h, w))
        if boxes is not None:
            for box_idx in range(len(boxes)):
                box = boxes[box_idx]
                label = int(labels[box_idx])
                catergory_info = {'id': label, 'is_crowd': 0}
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                annotation_info = pycococreatortools.create_annotation_info(
                        train_instance_cnt,
                        train_cnt,
                        catergory_info,
                        image_size=(w, h),
                        tolerance=0,
                        bounding_box=box
                )
                if annotation_info is not None:
                    train_o['annotations'].append(annotation_info)
                    train_instance_cnt += 1
            train_o['images'].append(image_info)
            train_cnt += 1
        elif not args.tb_only:
            train_o['images'].append(image_info)
            train_cnt += 1
    if not args.tb_only:
        with open(args.anno_savedir + os.path.basename(args.list_path[:-3]) + 'json', 'w') as f:
            json.dump(train_o, f)
            f.close()
    else:
        with open(args.anno_savedir + os.path.basename(args.list_path[:-4]) + '_only_tb.json', 'w') as f:
            json.dump(train_o, f)
            f.close()
    print('\nProcessing {} finished\n'.format(args.list_path))


if __name__ == '__main__':
    main()
