# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import glob
import PIL
import torch
from io import BytesIO
from PIL import Image
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageNet1k_JPG(torch.utils.data.Dataset):
  '''
  An ImageNet-1k dataset with caching support.
  '''

  def __init__(self, image_root, meta_path, transform):
    self.transform = transform

    with open(meta_path) as f:
      self.data_list = f.read().splitlines()
    self.image_root = image_root

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    line = self.data_list[idx]
    path, label = line.split(' ')

    path = os.path.join(self.image_root, path)
    label = int(label)

    image = Image.open(path).convert('RGB')
    image = self.transform(image)

    return image, label

def build_dataset_jpg(is_train, args):
    transform = build_transform(is_train, args)
    data_root = args.data_path
    image_root = os.path.join(data_root, 'train' if is_train else 'val')
    meta_path = os.path.join(data_root, 'meta', 'train.txt' if is_train else 'val.txt')
    dataset = ImageNet1k_JPG(image_root, meta_path, transform)
    print(f"Dataset at {meta_path}. Length of {len(dataset)}")
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

