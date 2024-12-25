import argparse
import os
import sys

import numpy as np
from PIL import Image
from torchvision.transforms import transforms

os.chdir(sys.path[0])
import torch
from torch.utils import data
from model.trans4passplus import Trans4PASSOneShareBackboneMoE
from dataset.stanford_pan8_dataset import StanfordPan8TestDataSet
from compute_iou import fast_hist, per_class_iu

import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

TARGET_NAME = 'span8'
EMB_CHANS = 128
DATA_DIRECTORY = '/home/jjiang/datasets/Stanford2D3D'
DATA_LIST_PATH = 'dataset/s2d3d_pan_list/val.txt'
INPUT_SIZE_TARGET = '2048,1024'

IGNORE_LABEL = 255
NUM_CLASSES = 8
SET = 'val'

AUX_RATE = 0.5

NAME_CLASSES = ['ceiling', 'chair', 'door', 'floor', 'sofa', 'table', 'wall', 'window']

colors = np.load('dataset/s2d3d_pin_list/colors.npy').tolist()  # for visualization
_key = [255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 5, 6, 7]
palette = []
for i, c in enumerate(colors):
    if _key[i] == 255:
        continue
    palette += c
zero_pad = 256 * 3 - len(palette)
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
    targettestset = StanfordPan8TestDataSet(args.data_dir, args.data_list, crop_size=(w, h),
                                            scale=False, mirror=False, set='val')
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)

    hist = np.zeros((args.num_classes, args.num_classes))
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print(f'[{index}/{len(testloader)}] processd')
        image, label, _, name = batch
        image = image.cuda(gpu0)
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
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), args.num_classes)

        if args.save_vis:
            output_col = colorize_mask(output)
            name = name[0].split('/')[-1]
            name = name.split(".")[0]
            # --- crop top, bottom black area
            output_col = output_col.resize((4096, 2048), Image.NEAREST)
            width, height = output_col.size
            left, top, right, bottom = 0, 320, width, 1728
            output_col = output_col.crop((left, top, right, bottom))

            output_col.save(f'{colored_pred_save_dir}/{name}color.png')

    mIoUs = per_class_iu(hist)
    for ind_class in range(args.num_classes):
        print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))


if __name__ == '__main__':
    main()
