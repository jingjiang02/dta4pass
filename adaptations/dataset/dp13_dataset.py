import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils.transform import RandomCropWH, FixScaleRandomCropWHBoth, GaussianBlur, FixScaleCropWH_Center


class densepass13DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 200),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', ssl_dir='',
                 trans='resize', need_grey=False, normalize=True, crop_pad=None):
        self.normalize = normalize
        self.need_grey = need_grey
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.trans = trans
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
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

        crop = RandomCropWH(image.size, self.crop_size)

        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
        elif self.trans == 'FixScaleRandomCropWH':
            # resize, keep ratio
            image = crop(image)
        else:
            raise NotImplementedError

        fix_scale_random_crop = FixScaleRandomCropWHBoth(image.size, (1024, 512))
        if self.need_grey:
            image_grey = fix_scale_random_crop(image_grey)
        image_cropped = fix_scale_random_crop(image)

        size = np.asarray(image, np.float32).shape

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
        image_cropped = input_transform(image_cropped)

        if len(self.ssl_dir) > 0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.jpg', '_labelTrainIds.png')))
            if self.trans == 'resize':
                # resize
                label = label.resize(self.crop_size, Image.NEAREST)
            elif self.trans == 'FixScaleRandomCropWH':
                # resize, keep ratio
                label = crop(label)
            else:
                raise NotImplementedError

            label = fix_scale_random_crop(label, is_label=True)
            label = torch.LongTensor(np.array(label).astype('int32'))
            if self.need_grey:
                return image, image_cropped, image_grey, np.array(size), name
            else:
                return image, image_cropped, label, np.array(size), name

        if self.need_grey:
            return image, image_cropped, image_grey, np.array(size), name
        else:
            return image, image_cropped, np.array(size), name


class densepass13DataSetWeakStrong(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 200),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', ssl_dir='',
                 trans='resize', need_grey=False, normalize=True, strong_aug_list=None, crop_pad=None):
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
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.trans = trans
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            self.files.append({
                "img": img_file,
                "name": name
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

        crop = RandomCropWH(image.size, self.crop_size)

        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
        elif self.trans == 'FixScaleRandomCropWH':
            # resize, keep ratio
            image = crop(image)
        else:
            raise NotImplementedError

        fix_scale_random_crop = FixScaleRandomCropWHBoth(image.size, (1024, 512))
        if self.need_grey:
            image_grey = fix_scale_random_crop(image_grey)
        image_cropped = fix_scale_random_crop(image)

        # strong aug
        image_strong = image.copy()
        image_cropped_strong = image_cropped.copy()

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
        image_cropped_strong = input_transform_strong(image_cropped_strong)

        size = np.asarray(image, np.float32).shape

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
        image_cropped = input_transform(image_cropped)
        image_strong = input_transform(image_strong)
        image_cropped_strong = input_transform(image_cropped_strong)

        if len(self.ssl_dir) > 0:
            label = Image.open(osp.join(self.ssl_dir, name.replace('.jpg', '_labelTrainIds.png')))
            if self.trans == 'resize':
                # resize
                label = label.resize(self.crop_size, Image.NEAREST)
            elif self.trans == 'FixScaleRandomCropWH':
                # resize, keep ratio
                label = crop(label)
            else:
                raise NotImplementedError
            label = fix_scale_random_crop(label, is_label=True)

            label = torch.LongTensor(np.array(label).astype('int32'))
            if self.need_grey:
                return image, image_strong, image_cropped, image_cropped_strong, image_grey, np.array(size), name
            else:
                return image, image_strong, image_cropped, image_cropped_strong, label, np.array(size), name

        if self.need_grey:
            return image, image_strong, image_cropped, image_cropped_strong, image_grey, np.array(size), name
        else:
            return image, image_strong, image_cropped, image_cropped_strong, np.array(size), name


class densepass13TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val', normalize=True, is_center_crop=False):
        self.is_center_crop = is_center_crop
        self.normalize = normalize
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace("_.png", "_labelTrainIds.png")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12])

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        # new_mask -= 1
        for value in values:
            if value == 255 or value <= -1:
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        mask = new_mask
        return mask

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        if self.is_center_crop:
            crop = FixScaleCropWH_Center(self.crop_size)
            image = crop(image)
            label = crop(label, is_label=True)
        else:
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        size = np.asarray(image).shape

        input_transform_list = [transforms.ToTensor()]
        if self.normalize:
            input_transform_list.append(transforms.Normalize((.485, .456, .406), (.229, .224, .225)))
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        return image, label, np.array(size), name


class densepass13TestDataSetGrey(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val', normalize=True):
        self.normalize = normalize
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace("_.png", "_labelTrainIds.png")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12])

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        # new_mask -= 1
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
        image_grey = image.convert('L')
        image = image.convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.asarray(image).shape

        fix_scale_random_crop = FixScaleRandomCropWHBoth(image.size, (1024, 512))
        image_cropped = fix_scale_random_crop(image)
        image_grey = fix_scale_random_crop(image_grey)

        input_transform_list = [transforms.ToTensor()]
        if self.normalize:
            input_transform_list.append(transforms.Normalize((.485, .456, .406), (.229, .224, .225)))
        input_transform = transforms.Compose(input_transform_list)

        image = input_transform(image)
        image_cropped = input_transform(image_cropped)
        input_transform_grey = transforms.Compose([
            transforms.ToTensor()
        ])
        image_grey = input_transform_grey(image_grey)

        label = torch.LongTensor(np.array(label).astype('int32'))
        return image, image_cropped, image_grey, label, np.array(size), name


if __name__ == '__main__':
    dst = densepassTestDataSet("data/DensePASS_train_pseudo_val", 'dataset/densepass_list/val.txt', mean=(0, 0, 0))
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
