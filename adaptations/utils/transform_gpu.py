import random

import kornia
import numpy as np
import torch
import torch.nn as nn


def colorJitter(colorJitter, data=None, target=None, s=0.25):
    # s is the strength of colorjitter
    # colorJitter
    if not (data is None):
        if data.shape[1] == 3:
            if colorJitter > 0.2:
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s))
                data = seq(data)
    data = torch.clamp(data, 0, 1)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    data = torch.clamp(data, 0, 1)
    return data, target


def flip(flip, data=None, target=None):
    # Flip
    if flip == 1:
        if not (data is None):
            data = torch.flip(data, (3,))  # np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
            data = torch.clamp(data, 0, 1)
        if not (target is None):
            target = torch.flip(target,
                                (2,))  # np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target


def cowMix(mask, data=None, target=None):
    # Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2] = 1 - stackedMask[1::2]
        data = (stackedMask * torch.cat((data[::2], data[::2])) + (1 - stackedMask) * torch.cat(
            (data[1::2], data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2] = 1 - stackedMask[1::2]
        target = (stackedMask * torch.cat((target[::2], target[::2])) + (1 - stackedMask) * torch.cat(
            (target[1::2], target[1::2]))).float()
    data = torch.clamp(data, 0, 1)
    return data, target


def mix(mask, data=None, target=None):
    # Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in
                              range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in
                                         range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in
                                         range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat(
            [(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in
             range(target.shape[0])])
    data = torch.clamp(data, 0, 1)
    return data, target


def oneMix(mask, data=None, target=None):
    # Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(0)
    data = torch.clamp(data, 0, 1)
    return data, target


def normalize(MEAN, STD, data=None, target=None):
    # Normalize
    if not (data is None):
        if data.shape[1] == 3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data - MEAN) / STD).float()
    data = torch.clamp(data, 0, 1)
    return data, target


def strong_aug(images, labels, weights=None):
    # strong aug
    images, labels = colorJitter(colorJitter=random.uniform(0, 1),
                                 data=images,
                                 target=labels,
                                 s=random.uniform(0.1, 0.3))
    images, labels = gaussian_blur(blur=random.uniform(0, 1),
                                   data=images,
                                   target=labels)
    is_flip = random.randint(0, 1)
    images, labels = flip(flip=is_flip,
                          data=images,
                          target=labels)
    if weights is not None:
        _, weights = flip(flip=is_flip, target=weights)
        return images, labels, weights
    else:

        return images, labels
