"""
    ARShadowGAN
    func  : 数据加载工具
    Author: Chen Yu
    Date  : 2020.10.20
"""
import os
import numpy as np
from PIL import Image
import random


DATA_DIR = 'Shadow-AR'
VS_DIR = os.path.join(DATA_DIR, 'shadow')
NO_VS_DIR = os.path.join(DATA_DIR, 'noshadow')
RS_MASK_DIR = os.path.join(DATA_DIR, 'rshadow')
RO_MASK_DIR = os.path.join(DATA_DIR, 'robject')
VO_MASK_DIR = os.path.join(DATA_DIR, 'mask')


def match_samples_path():
    samples = []
    for img in os.listdir(VS_DIR):
        if img in os.listdir(NO_VS_DIR) and img in os.listdir(RS_MASK_DIR) and \
                img in os.listdir(RO_MASK_DIR) and img in os.listdir(VO_MASK_DIR):
            samples.append({
                'vs': os.path.join(VS_DIR, img),
                'no_vs': os.path.join(NO_VS_DIR, img),
                'rs_mask': os.path.join(RS_MASK_DIR, img),
                'ro_mask': os.path.join(RO_MASK_DIR, img),
                'vo_mask': os.path.join(VO_MASK_DIR, img)
            })
    return samples


def _normalize(data, mean=0.5, std=0.5):
    data = (data - mean / std)
    return data


def augment_data(data):
    flip_direction = random.randint(0, 3)
    if flip_direction == 1:
        for k, v in data.items():
            data[k] = v.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_direction == 2:
        for k, v in data.items():
            data[k] = v.transpose(Image.FLIP_TOP_BOTTOM)
    # rotate_degree = random.randint(-45, 45)
    # for k, v in data.items():
    #     data[k] = v.rotate(rotate_degree)
    return data


def load_sample(sample, img_size=(256, 256)):
    data = {}
    for k, v in sample.items():
        if k == 'vs' or k == 'no_vs':
            data[k] = Image.open(v).convert("RGB").resize(img_size)
        else:
            data[k] = Image.open(v).convert("L").resize(img_size)
    data = augment_data(data)
    for k, v in data.items():
        data[k] = _normalize(np.asarray(v) / 255)
    return data





