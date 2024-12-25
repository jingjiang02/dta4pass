import logging
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils import data
from torchvision import transforms

from .seg_data_base import SegmentationDataset


class DensePASS13Segmentation(SegmentationDataset):
    """DensePASS Semantic Segmentation Dataset."""
    NUM_CLASS = 13

    def __init__(self, root='/home/jjiang/datasets/DensePASS/DensePASS', split='val',
                 mode=None, transform=None, **kwargs):
        super(DensePASS13Segmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/DensePASS"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        if self.mode != 'train':
            self.crop_size = [400, 2048]  # for inference only
            assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, -1, 12, 12])

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.zeros_like(mask)
        new_mask -= 1
        for value in values:
            if value == 255 or value <= -1:
                new_mask[mask == value] = -1
            else:
                new_mask[mask == value] = self._key[value]
        mask = new_mask
        return mask

    def _val_sync_transform_resize(self, img, mask):
        w, h = img.size
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        if self.mode == 'train':
            # fake mask, no mask for train
            mask = Image.open(
                '/home/jjiang/datasets/DensePASS/DensePASS/gtFine/val/cs/1_labelTrainIds.png')
        else:
            mask = Image.open(self.mask_paths[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._map19to13(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car')


class DensePASS13SegmentationForMorph(SegmentationDataset):
    """DensePASS Semantic Segmentation Dataset."""
    NUM_CLASS = 13

    def __init__(self, root='/home/jjiang/datasets/DensePASS/DensePASS', split='val',
                 mode=None, transform=None, trans='resize_crop', **kwargs):
        super(DensePASS13SegmentationForMorph, self).__init__(root, split, mode, transform, **kwargs)
        self.trans = trans
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/DensePASS"
        self.images, _ = _get_city_pairs(self.root, self.split)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        img = image.convert('RGB')
        image_grey = image.convert('L')

        if self.mode == 'train':
            img, image_grey = self._sync_transform_all(img, image_grey)
        else:
            raise Exception

        # strong aug
        img_strong = img.copy()

        input_transform_list_strong = []
        strong_aug_list = ['color jittering', 'grayscale', 'Gaussian blur', 'cutout patches']
        for strong_aug_name in strong_aug_list:
            if strong_aug_name == 'color jittering':
                input_transform_list_strong.append(
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                )
            elif strong_aug_name == 'grayscale':
                input_transform_list_strong.append(
                    transforms.RandomGrayscale(p=0.2)
                )
            elif strong_aug_name == 'Gaussian blur':
                input_transform_list_strong.append(
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5)
                )
            elif strong_aug_name == 'cutout patches':
                randcut_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.RandomErasing(
                            p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                        ),
                        transforms.RandomErasing(
                            p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                        ),
                        transforms.RandomErasing(
                            p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                        ),
                        transforms.ToPILImage(),
                    ]
                )
                input_transform_list_strong.append(
                    randcut_transform
                )
            else:
                raise Exception
        input_transform_strong = transforms.Compose(input_transform_list_strong)
        img_strong = input_transform_strong(img_strong)

        # convert to tensor
        input_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = input_transform(img)
        img_strong = input_transform(img_strong)
        image_grey = input_transform(image_grey)

        return img, img_strong, image_grey, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car')


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('_.png', '_labelTrainIds.png')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logging.info('cannot find the mask or image:', imgpath, maskpath)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    def get_path(img_folder):
        img_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.startswith('._'):
                    continue
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root, filename)
                    if os.path.isfile(imgpath):
                        img_paths.append(imgpath)
                    else:
                        logging.info('cannot find the image:', imgpath)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, []

    if split == 'train':
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        img_paths, mask_paths = get_path(img_folder)
        return img_paths, mask_paths
    else:
        logging.info('test set, but only val set')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        img_paths, mask_paths = get_path_pairs(val_img_folder, val_mask_folder)

    return img_paths, mask_paths


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


if __name__ == '__main__':
    dst = DensePASSSegmentation(split='train', mode='train')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        break
