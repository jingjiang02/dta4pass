import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils.transform import RandomHorizontalFlip, GaussianBlur


class synpass13DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255, set='val', need_grey=False, normalize=True, trans='resize',
                 flip=True):
        self.flip = flip
        self.normalize = normalize
        self.need_grey = need_grey
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            lbname = name.replace("img", "semantic").replace('.jpg', '_trainID.png')
            label_file = osp.join(self.root, lbname)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10, 255, 255, 255, 255, 6, 255, 255, 255, 9])

    def __len__(self):
        return len(self.files)

    def _map23to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255 or value <= -1:
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        mask = new_mask
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        if self.need_grey:
            image_grey = image.convert('L')
        image = image.convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map23to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)

        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        if self.need_grey:
            image_grey = image_grey.resize(self.crop_size, Image.BICUBIC)

        # other transformation
        if self.flip:
            transformations = [
                RandomHorizontalFlip()
            ]
            # 1. xxx
            for transformation in transformations:
                image = transformation(image)
                label = transformation(label, is_label=True)
                if self.need_grey:
                    image_grey = transformation(image_grey)

        size = np.array(image).shape

        input_transform_list = [transforms.ToTensor()]
        if self.normalize:
            input_transform_list.append(transforms.Normalize((.485, .456, .406), (.229, .224, .225)))
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)

        if self.need_grey:
            input_transform_grey = transforms.Compose([
                transforms.ToTensor()
            ])
            image_grey = input_transform_grey(image_grey)

        label = torch.LongTensor(np.array(label).astype('int32'))

        if self.need_grey:
            return image, image_grey, label, np.array(size), name
        else:
            return image, label, np.array(size), name


class synpass13DataSetWeakStrong(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128),
                 scale=True, mirror=True, ignore_label=255, set='val', need_grey=False, normalize=True,
                 strong_aug_list=None, flip=True):
        self.flip = flip
        if strong_aug_list is None:
            strong_aug_list = ['color jittering', 'grayscale', 'Gaussian blur']
        self.strong_aug_list = strong_aug_list
        self.normalize = normalize
        self.need_grey = need_grey
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            lbname = name.replace("img", "semantic").replace('.jpg', '_trainID.png')
            label_file = osp.join(self.root, lbname)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10, 255, 255, 255, 255, 6, 255, 255, 255, 9])

    def __len__(self):
        return len(self.files)

    def _map23to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255 or value <= -1:
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        mask = new_mask
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        if self.need_grey:
            image_grey = image.convert('L')
        image = image.convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map23to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)

        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        if self.need_grey:
            image_grey = image_grey.resize(self.crop_size, Image.BICUBIC)

        if self.flip:
            # other transformation
            transformations = [
                RandomHorizontalFlip()
            ]
            # 1. xxx
            for transformation in transformations:
                image = transformation(image)
                label = transformation(label, is_label=True)
                if self.need_grey:
                    image_grey = transformation(image_grey)

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

        size = np.array(image).shape

        input_transform_list = [transforms.ToTensor()]
        if self.normalize:
            input_transform_list.append(transforms.Normalize((.485, .456, .406), (.229, .224, .225)))
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)
        image_strong = input_transform(image_strong)

        if self.need_grey:
            input_transform_grey = transforms.Compose([
                transforms.ToTensor()
            ])
            image_grey = input_transform_grey(image_grey)

        label = torch.LongTensor(np.array(label).astype('int32'))

        if self.need_grey:
            return image, image_strong, image_grey, label, np.array(size), name
        else:
            return image, image_strong, label, np.array(size), name


if __name__ == '__main__':
    dst = synpassTestDataSet("data/SynPASS", 'dataset/SynPASS_list/val.txt', mean=(0, 0, 0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
            img.show()
        break
