import glob
import json
import os
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from dataset.stanford_pin8_dataset import __FOLD__
from torch.utils import data
from torchvision import transforms
from utils.transform import GaussianBlur


class StanfordPan8DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 1024),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0,
                 set='val', ssl_dir='', trans='resize', fold=1, need_grey=False, crop_pad=True, normalize=False,
                 org_mapping=False):
        self.crop_pad = crop_pad
        self.need_grey = need_grey
        self.root = root
        # self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        if not max_iters == None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        self.set = set
        self.trans = trans
        for p in self.img_paths:
            self.files.append({
                "img": p,
                "name": p.split(self.root + '/')[-1]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        if self.need_grey:
            image_grey = image.convert('L')
        image = image.convert('RGB')
        name = datafiles["name"]
        # --- crop top, bottom black area
        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728

        if self.crop_pad:
            image = image.crop((left, top, right, bottom))
            image = image.resize((width, height), Image.BILINEAR)
            if self.need_grey:
                image_grey = image_grey.crop((left, top, right, bottom))
                image_grey = image_grey.resize((width, height), Image.BILINEAR)

        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            if self.need_grey:
                image_grey = image_grey.resize(self.crop_size, Image.BICUBIC)
        else:
            raise NotImplementedError

        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = input_transform(image)

        if self.need_grey:
            input_transform_grey = transforms.Compose([
                transforms.ToTensor()
            ])
            image_grey = input_transform_grey(image_grey)

        if len(self.ssl_dir) > 0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.png', '_labelTrainIds.png')))

            # resize before crop, due to label size is [2048, 1024]
            label = label.resize((4096, 2048), Image.NEAREST)

            if self.crop_pad:
                label = label.crop((left, top, right, bottom))
                label = label.resize((width, height), Image.NEAREST)

            if self.trans == 'resize':
                # resize
                label = label.resize(self.crop_size, Image.NEAREST)
            else:
                raise NotImplementedError
            label = torch.LongTensor(np.array(label).astype('int32'))
            if self.need_grey:
                return image, image_grey, label, np.array(size), name
            else:
                return image, label, np.array(size), name

        if self.need_grey:
            return image, image_grey, np.array(size), name
        else:
            return image, np.array(size), name


class StanfordPan8DataSetWeakStrong(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 1024),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0,
                 set='val', ssl_dir='', trans='resize', fold=1, need_grey=False, crop_pad=True, strong_aug_list=None,
                 normalize=False, org_mapping=False):
        if strong_aug_list is None:
            strong_aug_list = ['color jittering', 'grayscale', 'Gaussian blur']
        self.strong_aug_list = strong_aug_list
        self.crop_pad = crop_pad
        self.need_grey = need_grey
        self.root = root
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        if not max_iters == None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        self.set = set
        self.trans = trans
        for p in self.img_paths:
            self.files.append({
                "img": p,
                "name": p.split(self.root + '/')[-1]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        if self.need_grey:
            image_grey = image.convert('L')
        image = image.convert('RGB')
        name = datafiles["name"]
        # --- crop top, bottom black area
        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728

        if self.crop_pad:
            image = image.crop((left, top, right, bottom))
            image = image.resize((width, height), Image.BILINEAR)
            if self.need_grey:
                image_grey = image_grey.crop((left, top, right, bottom))
                image_grey = image_grey.resize((width, height), Image.BILINEAR)

        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            if self.need_grey:
                image_grey = image_grey.resize(self.crop_size, Image.BICUBIC)
        else:
            raise NotImplementedError

        # strong aug
        image_strong = image.copy()

        input_transform_list_strong = []
        for strong_aug_name in self.strong_aug_list:
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

        image_strong = input_transform_strong(image_strong)

        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = input_transform(image)
        if self.need_grey:
            input_transform_grey = transforms.Compose([
                transforms.ToTensor()
            ])
            image_grey = input_transform_grey(image_grey)
        image_strong = input_transform(image_strong)

        if len(self.ssl_dir) > 0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.png', '_labelTrainIds.png')))

            # resize before crop, due to label size is [2048, 1024]
            label = label.resize((4096, 2048), Image.NEAREST)

            if self.crop_pad:
                label = label.crop((left, top, right, bottom))
                label = label.resize((width, height), Image.NEAREST)

            if self.trans == 'resize':
                # resize
                label = label.resize(self.crop_size, Image.NEAREST)
            else:
                raise NotImplementedError
            label = torch.LongTensor(np.array(label).astype('int32'))
            if self.need_grey:
                return image, image_strong, image_grey, label, np.array(size), name
            else:
                return image, image_strong, label, np.array(size), name

        if self.need_grey:
            return image, image_strong, image_grey, np.array(size), name
        else:
            return image, image_strong, np.array(size), name


class StanfordPan8TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 1024), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val', fold=1, trans='resize', normalize=False,
                 org_mapping=False):
        self.org_mapping = org_mapping
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)

        if not max_iters == None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        # --- stanford color2id
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy')  # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        self.trans = trans

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root + '/')[-1]
            })

        if self.org_mapping:
            self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        else:
            self._key = np.array([255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255 or value <= -1:
                mask[mask == value] = 255
            else:
                mask[mask == value] = self._key[value]
        return mask

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1  # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')

        label = Image.open(datafiles["label"])
        label = self._color2id(image, label)
        label = self._map13to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        # resize
        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            raise NotImplementedError

        size = np.asarray(image).shape

        input_transform_list = [transforms.ToTensor()]
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        # print(image.shape, label.shape)

        return image, label, np.array(size), name


class StanfordPan8TestDataSetGrey(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 1024), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val', fold=1, trans='resize', normalize=False,
                 org_mapping=False):
        self.org_mapping = org_mapping
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)

        if not max_iters == None:
            self.img_paths = self.img_paths * int(np.ceil(float(max_iters) / len(self.img_paths)))
        self.files = []
        # --- stanford color2id
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy')  # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        self.trans = trans

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root + '/')[-1]
            })

        if self.org_mapping:
            self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        else:
            self._key = np.array([255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255 or value <= -1:
                mask[mask == value] = 255
            else:
                mask[mask == value] = self._key[value]
        return mask

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1  # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        image_grey = image.convert('L')
        image = image.convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._color2id(image, label)
        label = self._map13to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        # --- crop top, bottom black area
        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728

        image = image.crop((left, top, right, bottom))
        image = image.resize((width, height), Image.BILINEAR)

        image_grey = image_grey.crop((left, top, right, bottom))
        image_grey = image_grey.resize((width, height), Image.BILINEAR)

        label = label.crop((left, top, right, bottom))
        label = label.resize((width, height), Image.NEAREST)
        # --- crop top, bottom black area done

        # resize
        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            image_cropped = image.resize((1024, 512), Image.BICUBIC)
            image_grey = image_grey.resize((1024, 512), Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            raise NotImplementedError

        size = np.asarray(image).shape

        input_transform_list = [transforms.ToTensor()]
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)
        image_cropped = input_transform(image_cropped)
        image_grey = input_transform(image_grey)
        label = torch.LongTensor(np.array(label).astype('int32'))

        return image, image_cropped, image_grey, label, np.array(size), name


def _get_stanford2d3d_path(folder, fold, mode='train'):
    '''image is jpg, label is png'''
    img_paths = []
    if mode == 'train':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'val':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'trainval':
        area_ids = __FOLD__[mode]
    else:
        raise NotImplementedError
    for a in area_ids:
        img_paths += glob.glob(os.path.join(folder, '{}/pano/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    return img_paths


if __name__ == '__main__':
    dst = StanfordPan8DataSet("data/Stanford2D3D", 'dataset/s2d3d_pan_list/val.txt', mean=(0, 0, 0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, _ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
            img.show()
        break
