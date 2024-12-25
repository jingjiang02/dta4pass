import argparse
import os
import sys

import numpy as np

from segmentron.utils.score import SegmentationMetric

os.chdir(sys.path[0])
import torch
from torch.utils import data
from model.trans4passplus import Trans4PASSOneShareBackboneMoE
from dataset.dp13_dataset import densepass13TestDataSet
from torchvision import transforms
from compute_iou import fast_hist, per_class_iu

from PIL import Image

import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

TARGET_NAME = 'DP13'
EMB_CHANS = 128
DATA_DIRECTORY = '/home/jjiang/datasets/DensePASS/DensePASS'
DATA_LIST_PATH = 'dataset/densepass_list/val.txt'
INPUT_SIZE_TARGET = '2048,400'

IGNORE_LABEL = 255
NUM_CLASSES = 13
SET = 'val'

AUX_RATE = 0.5

palette = [128, 64, 128,
           244, 35, 232,
           70, 70, 70,
           102, 102, 156,
           190, 153, 153,
           153, 153, 153,
           250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           152, 251, 152,
           70, 130, 180,
           220, 20, 60,
           # 255, 0, 0,
           0, 0, 142,
           # 0, 0, 70,
           # 0, 60, 100,
           # 0, 80, 100,
           # 0, 0, 230,
           # 119, 11, 32
           ]
zero_pad = 256 * 3 - len(palette)

NAME_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car']

for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--multi-scale", action='store_true')
    parser.add_argument("--no-norm", action='store_true')
    parser.add_argument("--save-vis", action='store_true')
    parser.add_argument("--name", type=str, default='DTA4PASS',
                        help="model_name")
    parser.add_argument("--fusion", type=str, default='fusion', choices=['fusion', 'moe', 'pin', 'pan'])
    return parser.parse_args()


def main():
    args = get_arguments()

    colored_pred_save_dir = f'{"/".join(args.restore_from.split("/")[:-1])}/colored_pred'
    if args.save_vis and not os.path.exists(colored_pred_save_dir):
        os.makedirs(colored_pred_save_dir)

    gpu0 = args.gpu
    model_name = args.name.lower()

    if model_name == 'dta4pass':
        norm = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
        model = Trans4PASSOneShareBackboneMoE(num_classes=NUM_CLASSES, emb_chans=EMB_CHANS, num_source_domains=2,
                                              norm=norm)
        args.no_norm = True
    else:
        raise Exception

    saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    print(msg)

    model.eval()
    model.cuda(gpu0)

    w, h = map(int, args.input_size_target.split(','))
    targettestset = densepass13TestDataSet(args.data_dir, args.data_list, crop_size=(w, h),
                                           mean=IMG_MEAN,
                                           scale=False, mirror=False, set='val', normalize=not args.no_norm)
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)
    hist = np.zeros((args.num_classes, args.num_classes))
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print(f'[{index}/{len(testloader)}] processd')
        image, label, _, name = batch
        image = image.cuda(gpu0)
        label = label.cuda(gpu0)
        b, _, _, _ = image.shape
        output_temp = torch.zeros((b, NUM_CLASSES, h, w), dtype=image.dtype).cuda(gpu0)
        if args.multi_scale:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # ms
        else:
            scales = [1]  # origin no scale
        for sc in scales:
            new_h, new_w = int(sc * h), int(sc * w)
            img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
            with torch.no_grad():
                # get output
                if model_name == 'dta4pass':
                    outputs, output = model(img_tem)
                    if args.fusion == 'fusion':
                        t = outputs + [output]
                        pred = sum(t) / len(t)
                    elif args.fusion == 'moe':
                        pred = output
                    elif args.fusion == 'pin':
                        pred = outputs[0]
                    elif args.fusion == 'pan':
                        pred = outputs[1]
                    else:
                        raise Exception

                output_temp += interp(pred)
        output = output_temp / len(scales)

        metric = SegmentationMetric(NUM_CLASSES, False)
        metric.update(output, label)
        _, mIoU = metric.get(return_category_iou=False)

        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)

        if args.save_vis:
            output_col = colorize_mask(output)
            name = name[0].split('/')[-1]
            name = name.split(".")[0]
            output_col.save(f'{colored_pred_save_dir}/{name}color.png')

            mIoU = round(float(mIoU) * 100, 2)
            output_col.save(f'{colored_pred_save_dir}/{name}color_{mIoU}.png')

    mIoUs = per_class_iu(hist)
    for ind_class in range(args.num_classes):
        print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))


if __name__ == '__main__':
    main()
