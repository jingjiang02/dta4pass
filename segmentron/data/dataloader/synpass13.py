"""Prepare SynPASS13 dataset"""
import glob
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from .seg_data_base import SegmentationDataset


class SynPASS13Segmentation(SegmentationDataset):
    """SynPASS Semantic Segmentation Dataset."""
    NUM_CLASS = 13

    def __init__(self, root='/home/jjiang/datasets/SynPASS/SynPASS', split='val',
                 mode=None, transform=None, weather='all', **kwargs):
        super(SynPASS13Segmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/SynPASS"
        self.root = root
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        self._key = np.array([-1, 2, 4, -1, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10, -1, -1, -1, -1, 6, -1, -1, -1, 9])

    def _map23to13(self, mask):
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
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
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
        target = self._map23to13(np.array(mask).astype('int32'))
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


class SynPASS13SegmentationForMorph(SegmentationDataset):
    """SynPASS Semantic Segmentation Dataset."""
    NUM_CLASS = 13

    def __init__(self, root='/home/jjiang/datasets/SynPASS/SynPASS', split='val',
                 mode=None, transform=None, weather='all', trans='resize', **kwargs):
        super(SynPASS13SegmentationForMorph, self).__init__(root, split, mode, transform, **kwargs)
        self.trans = trans
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/SynPASS"
        self.root = root
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        self._key = np.array([-1, 2, 4, -1, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10, -1, -1, -1, -1, 6, -1, -1, -1, 9])

    def _map23to13(self, mask):
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
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        img = image.convert('RGB')
        image_grey = image.convert('L')
        mask = Image.open(self.mask_paths[index])

        if self.mode == 'train':
            img, image_grey, mask = self._sync_transform_all(img, image_grey, mask)
        else:
            raise Exception

        # convert to tensor
        input_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = input_transform(img)
        image_grey = input_transform(image_grey)
        mask = self._mask_transform(mask)

        return img, image_grey, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._map23to13(np.array(mask).astype('int32'))
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


def _get_city_pairs(folder, split='train'):
    # datasets/SynPASS/img/cloud/train/MAP_1_point2/000000.jpg
    img_paths = glob.glob(os.path.join(*[folder, 'img', '*', split, '*', '*.jpg']))
    img_paths = sorted(img_paths)
    # datasets/SynPASS/semantic/cloud/train/MAP_1_point2/000000_trainID.png
    mask_paths = glob.glob(os.path.join(*[folder, 'semantic', '*', split, '*', '*_trainID.png']))
    mask_paths = sorted(mask_paths)
    assert len(img_paths) == len(mask_paths)
    logging.info('Found {} images in the folder {}'.format(len(img_paths), folder))
    return img_paths, mask_paths


if __name__ == '__main__':
    dst = SynPASS13Segmentation(split='train', mode='train')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        break
