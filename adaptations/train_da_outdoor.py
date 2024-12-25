# release code
import argparse
import glob
import logging
import math
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms

from dataset.cs13_dataset_src import CS13SrcDataSet
from dataset.dp13_dataset import densepass13DataSetWeakStrong, densepass13TestDataSet
from dataset.sp13_dataset import synpass13DataSet
from model.class_mix import mix_source_target
from model.pseudo_label import get_pseudo_label_online
from model.stylegan_networks import StyleGAN2Discriminator, StyleGAN2DiscriminatorUNet
from model.trans4passplus import Trans4PASSOneShareBackboneMoE
from morph.TransMorph_diff_rgb import CONFIGS as CONFIGS_TM
from morph.TransMorph_diff_rgb import TransMorphDiffRGB, Bilinear
from utils import transform_gpu
from utils.color_transform import RGB_LAB_Converter
from utils.init import *
from utils.loss import UncertaintyLoss, BCEWithLogitsLossPixelWiseWeighted
from utils.transform import TensorFixScaleRandomCropWHBoth
from utils.val_util import MultiHeadEvaluator

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# model morph option
base_channel = 1  # 3 or 1
D_G_name = 'StyleGAN2DiscriminatorUnet'  # StyleGAN2DiscriminatorUnet；StyleGAN2Discriminator;
D_G_optimizer_name = 'RMSprop'  # RMSprop; Adam; SGD
LEARNING_RATE_D_MORPH = 2.5e-6  # origin 1e-4
LEARNING_RATE_D_MORPH_MAX = LEARNING_RATE_D_MORPH * 2
LEARNING_RATE_MORPH = 2.5e-6  # learning rate
LEARNING_RATE_MORPH_MAX = LEARNING_RATE_MORPH * 2
MORPH_SCHEDULER_TYPE = 'cos'  # cos; poly
morph_weight_decay = 5e-5  # default 1e-5
# model morph option done

# model seg option
MODEL = 'Trans4PASSOneShareBackboneMoE'
EMB_CHANS = 128
NUM_STEPS = 40000
WARM_UP_STEPS = int(NUM_STEPS * 0.2)  # warmup lr
NUM_STEPS_STOP = int(NUM_STEPS * 1.0)  # early stopping
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 250  # for example 10000->250
SAVE_IMG_PRED_EVERY = int(SAVE_PRED_EVERY * 10)
SAVE_CKPT_EVERY = NUM_STEPS  # NUM_STEPS:no_save; 20000
POWER = 0.9

model_optimizer_name = 'SGD'  # SGD; AdamW
pseudo_rate = 0.5
pseudo_threshold = 0.95
SCHEDULER_TYPE = 'poly'  # cos; poly
LEARNING_RATE = 5e-6
LEARNING_RATE_MAX = LEARNING_RATE * 10
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# model seg option done

# dataset and other option
NUM_DATASET = 2
ORIGIN_BATCH_SIZE = 2
BATCH_SIZE = ORIGIN_BATCH_SIZE // NUM_DATASET
NUM_WORKERS = BATCH_SIZE * 2
TARGET_NAME = 'DP13'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '/home/jjiang/datasets/DensePASS/DensePASS'
DATA_LIST_PATH_TARGET = 'dataset/densepass_list/train.txt'
DATA_LIST_PATH_TARGET_TEST = 'dataset/densepass_list/val.txt'
INPUT_SIZE_TARGET = '2048,400'
TARGET_TRANSFORM = 'FixScaleRandomCropWH'
INPUT_SIZE_TARGET_TEST = '2048,400'
TARGET = 'densepass13'
SET = 'train'
NUM_CLASSES = 13
RANDOM_SEED = 1234

# teacher setting
TEACHER_UPDATE_ITER = 1
EMA_KEEP_RATE = 0.999

# mix setting
mix_ignore = IGNORE_LABEL  # IGNORE_LABEL, None

NAME_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car']


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : Trans4PASS_v1, Trans4PASS_v2")
    parser.add_argument("--emb-chans", type=int, default=EMB_CHANS,
                        help="Number of channels in decoder head.")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default='',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default='',
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-target-test", type=str, default=DATA_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target val dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default='',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default='',
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--continue-train", action="store_true",
                        help="continue training")
    return parser.parse_args()


args = get_arguments()


def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logging.root.name = name
    logging.root.setLevel(logging.INFO)
    # don't log results for the non-master process
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.root.addHandler(fh)
    # else:
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def warmup_cosine_annealing_lr(current_step, total_steps, warmup_steps, base_lr, max_lr, end_lr):
    if current_step < warmup_steps:
        # 线性预热阶段
        warmup_lr = base_lr + (max_lr - base_lr) * current_step / warmup_steps
        return warmup_lr
    else:
        # 余弦退火阶段
        # 计算余弦退火阶段的步数
        annealing_steps = total_steps - warmup_steps
        # 计算当前在退火阶段的步数
        annealing_step = current_step - warmup_steps
        # 使用余弦退火公式调整学习率
        cosine_lr = end_lr + 0.5 * (max_lr - end_lr) * (1 + math.cos(math.pi * annealing_step / annealing_steps))
        return cosine_lr


def adjust_learning_rate(optimizer, i_iter):
    if SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE, i_iter, args.num_steps, POWER)
    elif SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE, LEARNING_RATE_MAX,
                                        LEARNING_RATE)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_morph(optimizer, i_iter):
    if MORPH_SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE_MORPH, i_iter, args.num_steps, POWER)
    elif MORPH_SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE_MORPH,
                                        LEARNING_RATE_MORPH_MAX,
                                        LEARNING_RATE_MORPH)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_D_morph(optimizer, i_iter):
    if MORPH_SCHEDULER_TYPE == 'poly':
        lr = lr_poly(LEARNING_RATE_D_MORPH, i_iter, args.num_steps, POWER)
    elif MORPH_SCHEDULER_TYPE == 'cos':
        lr = warmup_cosine_annealing_lr(i_iter, args.num_steps, WARM_UP_STEPS, LEARNING_RATE_D_MORPH,
                                        LEARNING_RATE_D_MORPH_MAX,
                                        LEARNING_RATE_D_MORPH)
    else:
        raise Exception
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


@torch.no_grad()
def _update_teacher_model(student_model, teacher_model, keep_rate=0.9996):
    student_model_dict = student_model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)


def main():
    """Create the model and start the training."""
    # set random seed
    set_random_seed(args.random_seed)

    # set para
    input_nc = base_channel

    RESTORE_FROM = '/home/jjiang/experiments/MultiMorph4PASS/final_ckpts_normed/outdoor/source_models/2024-07-21-07-39_best_dp13_model_54.363.pth'

    # change args
    PIN_DATA_DIRECTORY = '/home/jjiang/datasets/Cityscapes'
    PIN_DATA_LIST_PATH = 'dataset/cityscapes_list/train.txt'

    PAN_DATA_DIRECTORY = '/home/jjiang/datasets/SynPASS/SynPASS'
    PAN_DATA_LIST_PATH = 'dataset/synpass_list/train.txt'

    DIR_NAME = 'my_CS13SP13_{}_{}_Morph_WarmUp_'.format(TARGET_NAME, MODEL)
    SNAPSHOT_DIR = 'snapshots/' + DIR_NAME
    LOG_DIR = SNAPSHOT_DIR
    exp_name = args.snapshot_dir
    args.snapshot_dir = SNAPSHOT_DIR + exp_name
    args.log_dir = LOG_DIR + exp_name
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    setup_logger('Trans4PASS', args.log_dir, f'{TIME_STAMP}_log.txt')

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    w, h = map(int, INPUT_SIZE_TARGET_TEST.split(','))
    input_size_target_test = (w, h)

    Iter = 0
    bestIoU = 0
    mIoU = 0
    mIoU_pin = 0
    mIoU_pan = 0
    mIoU_moe = 0
    mIoU_teacher = 0

    norm = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    freeze_model(norm)
    # Create network
    # init G_pin
    model = Trans4PASSOneShareBackboneMoE(num_classes=args.num_classes, emb_chans=args.emb_chans, norm=norm)

    saved_state_dict = torch.load(RESTORE_FROM, map_location=lambda storage, loc: storage)
    if 'state_dict' in saved_state_dict.keys():
        saved_state_dict = saved_state_dict['state_dict']
    msg = model.load_state_dict(saved_state_dict, strict=False)
    logging.info(msg)

    # init G_pin teacher
    model_teacher = Trans4PASSOneShareBackboneMoE(num_classes=args.num_classes, emb_chans=args.emb_chans, norm=norm)

    msg = model_teacher.load_state_dict(saved_state_dict, strict=False)
    logging.info(msg)

    # init D_MORPH
    if D_G_name == 'StyleGAN2Discriminator':
        # TODO focus here
        loss_adv_target_rate = 5
        loss_smooth_rate = 0.25
        loss_recon_rate = 5
        loss_sem_rate = 0.5
        # loss rate hyper-para done

        model_D_morph_pin2pan = StyleGAN2Discriminator(input_nc=input_nc, size=1024).to(device)
        model_D_morph_source_pin2pan = StyleGAN2Discriminator(input_nc=input_nc, size=1024).to(device)
    elif D_G_name == 'StyleGAN2DiscriminatorUnet':
        # TODO focus here
        loss_adv_target_rate = 20.0
        loss_smooth_rate = 1.0
        loss_recon_rate = loss_smooth_rate
        loss_sem_rate = loss_smooth_rate

        scale_factor = 0.25
        # loss rate hyper-para done

        model_D_morph_pin2pan = StyleGAN2DiscriminatorUNet(input_nc=input_nc, size=1024, pixel_factor=scale_factor).to(
            device)
        model_D_morph_source_pin2pan = StyleGAN2DiscriminatorUNet(input_nc=input_nc, size=1024,
                                                                  pixel_factor=scale_factor).to(device)
    else:
        raise Exception

    unfreeze_model(model)
    model.to(device)
    # freeze model_teacher, update it using EMA
    freeze_model(model_teacher)
    model_teacher.to(device)

    unfreeze_model(model_D_morph_pin2pan)
    model_D_morph_pin2pan.to(device)

    unfreeze_model(model_D_morph_source_pin2pan)
    model_D_morph_source_pin2pan.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    else:
        pass

    # init data loader
    trainset_pin = CS13SrcDataSet(PIN_DATA_DIRECTORY, PIN_DATA_LIST_PATH,
                                  max_iters=args.num_steps * ORIGIN_BATCH_SIZE,
                                  crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                                  mean=IMG_MEAN, set=args.set, need_grey=base_channel == 1, normalize=False)
    trainloader_pin = data.DataLoader(trainset_pin, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    trainloader_iter_pin = enumerate(trainloader_pin)

    trainset_pan = synpass13DataSet(PAN_DATA_DIRECTORY, PAN_DATA_LIST_PATH,
                                    max_iters=args.num_steps * ORIGIN_BATCH_SIZE,
                                    crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                                    mean=IMG_MEAN, set=args.set, need_grey=base_channel == 1, normalize=False)
    trainloader_pan = data.DataLoader(trainset_pan, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    trainloader_iter_pan = enumerate(trainloader_pan)

    strong_aug_list = ['color jittering', 'Gaussian blur', 'cutout patches']
    targetset = densepass13DataSetWeakStrong(args.data_dir_target, args.data_list_target,
                                             max_iters=args.num_steps * ORIGIN_BATCH_SIZE,
                                             crop_size=input_size_target, scale=False, mirror=args.random_mirror,
                                             mean=IMG_MEAN,
                                             set=args.set,
                                             trans=TARGET_TRANSFORM, need_grey=base_channel == 1, normalize=False,
                                             strong_aug_list=strong_aug_list)
    targetloader = data.DataLoader(targetset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    logging.info('\n--- load TEST dataset ---')

    # test_h, test_w = 400, 2048
    test_w, test_h = input_size_target_test
    targettestset = densepass13TestDataSet(args.data_dir_target, args.data_list_target_test, crop_size=(test_w, test_h),
                                           mean=IMG_MEAN, scale=False, mirror=False, set='val', normalize=False)
    testloader = data.DataLoader(targettestset, batch_size=1, shuffle=False, pin_memory=True)

    # init optimizer
    if model_optimizer_name == 'SGD':
        optimizer_seg = optim.SGD(model.parameters(),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif model_optimizer_name == 'AdamW':
        optimizer_seg = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception
    optimizer_seg.zero_grad()

    if D_G_optimizer_name == 'Adam':
        optimizer_D_morph_pin2pan = optim.Adam(model_D_morph_pin2pan.parameters(), lr=LEARNING_RATE_D_MORPH,
                                               betas=(0.9, 0.99),
                                               weight_decay=morph_weight_decay)
        optimizer_D_morph_source_pin2pan = optim.Adam(model_D_morph_source_pin2pan.parameters(),
                                                      lr=LEARNING_RATE_D_MORPH, betas=(0.9, 0.99),
                                                      weight_decay=morph_weight_decay)
    elif D_G_optimizer_name == 'SGD':
        optimizer_D_morph_pin2pan = optim.SGD(model_D_morph_pin2pan.parameters(), lr=LEARNING_RATE_D_MORPH,
                                              momentum=args.momentum,
                                              weight_decay=morph_weight_decay)
        optimizer_D_morph_source_pin2pan = optim.SGD(model_D_morph_source_pin2pan.parameters(),
                                                     lr=LEARNING_RATE_D_MORPH, momentum=args.momentum,
                                                     weight_decay=morph_weight_decay)
    elif D_G_optimizer_name == 'RMSprop':
        optimizer_D_morph_pin2pan = optim.RMSprop(model_D_morph_pin2pan.parameters(),
                                                  lr=LEARNING_RATE_D_MORPH, momentum=args.momentum,
                                                  weight_decay=morph_weight_decay)
        optimizer_D_morph_source_pin2pan = optim.RMSprop(model_D_morph_source_pin2pan.parameters(),
                                                         lr=LEARNING_RATE_D_MORPH, momentum=args.momentum,
                                                         weight_decay=morph_weight_decay)
    else:
        raise Exception
    optimizer_D_morph_pin2pan.zero_grad()
    optimizer_D_morph_source_pin2pan.zero_grad()

    # init loss

    bce_loss = torch.nn.BCEWithLogitsLoss()
    bce_loss_with_weight = BCEWithLogitsLossPixelWiseWeighted()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    seg_loss_unreduced = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction='none')
    uncertainty_loss = UncertaintyLoss()

    # similarity loss
    sim_loss = torch.nn.L1Loss()
    # similarity loss done

    # labels for adversarial training
    source_label = 0
    target_label = 1
    eps = 0.1
    smoothed_source_label = source_label + eps
    smoothed_target_label = target_label - eps

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    # get deformable fild
    '''
    Initialize model
    '''
    model_name = 'transmorph'  # voxel_morph; transmorph
    if model_name == 'transmorph':
        config = CONFIGS_TM['TransMorphDiffRGB']
        model_morph = TransMorphDiffRGB(config, channel=base_channel,
                                        input_size=(input_size[1], input_size[0]))
    else:
        raise Exception
    model_morph.to(device)
    unfreeze_model(model_morph)

    optimizer_morph = optim.Adam(model_morph.parameters(), lr=LEARNING_RATE_MORPH, weight_decay=morph_weight_decay,
                                 amsgrad=True)
    optimizer_morph.zero_grad()

    '''
    Initialize spatial transformation function
    '''
    reg_model = Bilinear(zero_boundary=True, mode='nearest').cuda()
    freeze_model(reg_model)
    reg_model_bilin = Bilinear(zero_boundary=True, mode='bilinear').cuda()
    freeze_model(reg_model_bilin)

    grid_img = mk_grid_img(32, 1, (args.batch_size, input_size[1], input_size[0]))

    # draw
    plt.switch_backend('agg')
    converter = RGB_LAB_Converter(device)
    evaluator = MultiHeadEvaluator(testloader, NUM_CLASSES, NAME_CLASSES, device, softmax=False)
    # start training
    for i_iter in range(Iter, args.num_steps + 1):
        # reset optimizer
        optimizer_seg.zero_grad()
        lr_model = adjust_learning_rate(optimizer_seg, i_iter)

        optimizer_morph.zero_grad()
        lr_morph = adjust_learning_rate_morph(optimizer_morph, i_iter)

        optimizer_D_morph_pin2pan.zero_grad()
        lr_D_morph = adjust_learning_rate_D_morph(optimizer_D_morph_pin2pan, i_iter)

        optimizer_D_morph_source_pin2pan.zero_grad()
        lr_D_morph = adjust_learning_rate_D_morph(optimizer_D_morph_source_pin2pan, i_iter)

        # model freeze & unfreeze
        unfreeze_model(model)
        unfreeze_model(model_morph)
        unfreeze_model(model_D_morph_pin2pan)
        unfreeze_model(model_D_morph_source_pin2pan)

        # get data
        _, batch_source_pin = trainloader_iter_pin.__next__()
        images_source_pin_origin, images_source_pin_grey, labels_source_pin, _, _ = batch_source_pin
        images_source_pin_origin = images_source_pin_origin.to(device)
        images_source_pin_grey = images_source_pin_grey.to(device)
        labels_source_pin = labels_source_pin.long().to(device)

        _, batch_source_pan = trainloader_iter_pan.__next__()
        images_source_pan_origin, images_source_pan_grey, labels_source_pan, _, _ = batch_source_pan
        images_source_pan_origin = images_source_pan_origin.to(device)
        images_source_pan_grey = images_source_pan_grey.to(device)
        labels_source_pan = labels_source_pan.long().to(device)

        _, batch_target = targetloader_iter.__next__()
        images_target, images_target_strong, _, _, images_target_grey, _, _ = batch_target
        images_target = images_target.to(device)
        images_target_strong = images_target_strong.to(device)
        images_target_grey = images_target_grey.to(device)

        crop_opt = TensorFixScaleRandomCropWHBoth(input_size_target, input_size)
        images_target_cropped = crop_opt(images_target)

        # color convert
        images_source_pin = converter(images_source_pin_origin, images_target_cropped)
        images_source_pan = converter(images_source_pan_origin, images_target_cropped)

        # prepare done
        # warp with flow
        images_pin2pan_grey, flow_pin2pan, images_pan2pin_grey, flow_pan2pin = model_morph(
            (images_source_pin_grey, images_target_grey))
        loss_smooth = model_morph.scale_reg_loss()

        images_source_pin2pan_grey, flow_source_pin2pan, images_source_pan2pin_grey, flow_source_pan2pin = model_morph(
            (images_source_pin_grey, images_source_pan_grey))
        loss_smooth += model_morph.scale_reg_loss()

        # get images_source2target and images_target2source
        images_pin2pan = reg_model_bilin(images_source_pin.float(), flow_pin2pan)
        images_pan2pin = reg_model_bilin(images_target_cropped.float(), flow_pan2pin)
        images_source_pin2pan = reg_model_bilin(images_source_pin.float(), flow_source_pin2pan)
        images_source_pan2pin = reg_model_bilin(images_source_pan.float(), flow_source_pan2pin)

        # deal with label 0, due to reg_model may generate black edge, which value is also 0
        with torch.no_grad():
            labels_pin2pan = warp_labels(labels_source_pin, flow_pin2pan, reg_model, ignore_label=IGNORE_LABEL)
            labels_pin2pan = labels_pin2pan.detach()
            labels_source_pin2pan = warp_labels(labels_source_pin, flow_source_pin2pan, reg_model,
                                                ignore_label=IGNORE_LABEL)
            labels_source_pin2pan = labels_source_pin2pan.detach()
        # warp with flow done

        # ---step1 train model_seg---

        # get pseudo_labels for target & target cropped
        with torch.no_grad():
            pred_teacher_target = model_teacher(images_target, ensemble=True)
            pred_teacher_target = F.softmax(pred_teacher_target, dim=1)

            labels_target_ensemble = get_pseudo_label_online(pred_teacher_target, IGNORE_LABEL, device,
                                                             threshold=pseudo_threshold,
                                                             need_weight=False, softmax=False)

            # crop full image to cropped image
            labels_target_cropped_ensemble = crop_opt(labels_target_ensemble, is_label=True)
            labels_target_cropped_ensemble_weight = torch.ones_like(labels_target_cropped_ensemble).float().to(
                device)  # no weights

        # train target
        pred_target_all, pred_target = model(images_target_strong)

        loss_target = pseudo_rate * (
            uncertainty_loss(seg_loss_unreduced(pred_target, labels_target_ensemble), pred_target_all[0],
                             pred_target_all[1]))

        # train with mixed
        loss_mixed = 0.

        # class mix for pin2pan&target
        mixed_images_pin2pan, mixed_labels_pin2pan, _ = mix_source_target(images_pin2pan.detach(),
                                                                          labels_pin2pan,
                                                                          images_target_cropped,
                                                                          labels_target_cropped_ensemble,
                                                                          labels_target_cropped_ensemble_weight,
                                                                          ignore_label=mix_ignore)
        mixed_images_pin2pan, mixed_labels_pin2pan = transform_gpu.strong_aug(mixed_images_pin2pan,
                                                                              mixed_labels_pin2pan)

        pred_mixed_pin2pan_all, pred_mixed_pin2pan = model(mixed_images_pin2pan)

        loss_mixed += pseudo_rate * (
            uncertainty_loss(seg_loss_unreduced(pred_mixed_pin2pan, mixed_labels_pin2pan), pred_mixed_pin2pan_all[0],
                             pred_mixed_pin2pan_all[1])) / 2.0

        # class mix for source pin2pan&target
        mixed_images_source_pin2pan, mixed_labels_source_pin2pan, _ = mix_source_target(
            images_source_pin2pan.detach(),
            labels_source_pin2pan,
            images_target_cropped,
            labels_target_cropped_ensemble,
            labels_target_cropped_ensemble_weight,
            ignore_label=mix_ignore)
        mixed_images_source_pin2pan, mixed_labels_source_pin2pan = transform_gpu.strong_aug(mixed_images_source_pin2pan,
                                                                                            mixed_labels_source_pin2pan)
        pred_mixed_source_pin2pan_all, pred_mixed_source_pin2pan = model(mixed_images_source_pin2pan)

        loss_mixed += pseudo_rate * (
            uncertainty_loss(seg_loss_unreduced(pred_mixed_source_pin2pan, mixed_labels_source_pin2pan),
                             pred_mixed_source_pin2pan_all[0], pred_mixed_source_pin2pan_all[1])) / 2.0

        # class mix for pan&target
        mixed_images_pan, mixed_labels_pan, _ = mix_source_target(images_source_pan,
                                                                  labels_source_pan,
                                                                  images_target_cropped,
                                                                  labels_target_cropped_ensemble,
                                                                  labels_target_cropped_ensemble_weight,
                                                                  ignore_label=mix_ignore)
        mixed_images_pan, mixed_labels_pan = transform_gpu.strong_aug(mixed_images_pan, mixed_labels_pan)

        pred_mixed_pan_all, pred_mixed_pan = model(mixed_images_pan)

        loss_mixed += pseudo_rate * (
            uncertainty_loss(seg_loss_unreduced(pred_mixed_pan, mixed_labels_pan), pred_mixed_pan_all[0],
                             pred_mixed_pan_all[1]))

        # 1. L^seg
        pred_pin2pan, _ = model(images_pin2pan.detach())
        pred_pin2pan = pred_pin2pan[0]
        pred_source_pin2pan, _ = model(images_source_pin2pan.detach())
        pred_source_pin2pan = pred_source_pin2pan[0]
        # 2. L^col_S
        loss_pin = (seg_loss(pred_pin2pan, labels_pin2pan) + seg_loss(pred_source_pin2pan, labels_source_pin2pan)) / 2.0

        # cross with pin
        pred_pan, _ = model(images_source_pan)
        pred_pan = pred_pan[1]
        loss_pan = seg_loss(pred_pan, labels_source_pan)

        model_seg_total_loss = loss_mixed + loss_pin + loss_pan + loss_target
        model_seg_total_loss.backward()

        optimizer_seg.step()
        # ---step1 train model_seg done---

        # ---step2 train model_morph---
        # freeze model
        freeze_model(model)
        # train G
        freeze_model(model_D_morph_pin2pan)
        freeze_model(model_D_morph_source_pin2pan)

        # warp with flow
        images_pin_reconstruct = reg_model_bilin(images_pin2pan.float(), flow_pan2pin)
        images_pan_reconstruct = reg_model_bilin(images_pan2pin.float(), flow_pin2pan)
        images_source_pin_reconstruct = reg_model_bilin(images_source_pin2pan.float(), flow_source_pan2pin)
        images_source_pan_reconstruct = reg_model_bilin(images_source_pan2pin.float(), flow_source_pin2pan)
        loss_recon = sim_loss(images_source_pin, images_pin_reconstruct) \
                     + sim_loss(images_target_cropped, images_pan_reconstruct) \
                     + sim_loss(images_source_pin, images_source_pin_reconstruct) \
                     + sim_loss(images_source_pan, images_source_pan_reconstruct)
        loss_recon = loss_recon * loss_recon_rate

        loss_sem = sim_loss(model_teacher(images_pin2pan, ensemble=True),
                            reg_model_bilin(model_teacher(images_source_pin, ensemble=True), flow_pin2pan)) + \
                   sim_loss(model_teacher(images_pan2pin, ensemble=True),
                            reg_model_bilin(model_teacher(images_target_cropped, ensemble=True), flow_pan2pin)) + \
                   sim_loss(model_teacher(images_source_pin2pan, ensemble=True),
                            reg_model_bilin(model_teacher(images_source_pin, ensemble=True), flow_source_pin2pan)) + \
                   sim_loss(model_teacher(images_source_pan2pin, ensemble=True),
                            reg_model_bilin(model_teacher(images_source_pan, ensemble=True), flow_source_pan2pin))

        loss_sem = loss_sem * loss_sem_rate

        # D
        D_out_full_pin2pan, D_out = model_D_morph_pin2pan(images_pin2pan_grey)
        # D_out_full.shape=torch.Size([1, 1, 512, 1024])
        # D_out.shape=torch.Size([1, 1])
        loss_adv_target = bce_loss(D_out,
                                   torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device)) + \
                          bce_loss(D_out_full_pin2pan,
                                   torch.FloatTensor(D_out_full_pin2pan.data.size()).fill_(smoothed_target_label).to(
                                       device))
        loss_adv_target_all = loss_adv_target * loss_adv_target_rate

        D_out_full_source_pin2pan, D_out = model_D_morph_source_pin2pan(images_source_pin2pan_grey)
        loss_adv_target = bce_loss(D_out,
                                   torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device)) + \
                          bce_loss(D_out_full_source_pin2pan,
                                   torch.FloatTensor(D_out_full_source_pin2pan.data.size()).fill_(
                                       smoothed_target_label).to(
                                       device))
        loss_adv_target_all += loss_adv_target * loss_adv_target_rate

        # smooth
        loss_smooth = loss_smooth * loss_smooth_rate

        loss_morph = (i_iter / args.num_steps) * (
                loss_recon + loss_sem + loss_smooth) + loss_adv_target_all
        loss_morph.backward()

        # === train D G
        unfreeze_model(model_D_morph_pin2pan)
        unfreeze_model(model_D_morph_source_pin2pan)

        # train with source
        loss_D_value_morph = 0.
        images_pin2pan_grey = images_pin2pan_grey.detach()
        D_out_full, D_out = model_D_morph_pin2pan(images_pin2pan_grey)

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device)) + \
                 bce_loss(D_out_full, torch.FloatTensor(D_out_full.data.size()).fill_(source_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value_morph += loss_D.item()

        # train with target
        images_target_grey = images_target_grey.detach()
        D_out_full, D_out = model_D_morph_pin2pan(images_target_grey)

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device)) + \
                 bce_loss(D_out_full, torch.FloatTensor(D_out_full.data.size()).fill_(smoothed_target_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value_morph += loss_D.item()

        # train with source
        images_source_pin2pan_grey = images_source_pin2pan_grey.detach()
        D_out_full, D_out = model_D_morph_source_pin2pan(images_source_pin2pan_grey)

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device)) + \
                 bce_loss(D_out_full, torch.FloatTensor(D_out_full.data.size()).fill_(source_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value_morph += loss_D.item()

        # train with target
        images_source_pan_grey = images_source_pan_grey.detach()
        D_out_full, D_out = model_D_morph_source_pin2pan(images_source_pan_grey)

        loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(smoothed_target_label).to(device)) + \
                 bce_loss(D_out_full, torch.FloatTensor(D_out_full.data.size()).fill_(smoothed_target_label).to(device))
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value_morph += loss_D.item()

        optimizer_morph.step()
        optimizer_D_morph_pin2pan.step()
        optimizer_D_morph_source_pin2pan.step()
        # ---step2 train model_morph done---

        # update model teacher
        if i_iter % TEACHER_UPDATE_ITER == 0:
            # 则用EMA更新教师网络
            _update_teacher_model(student_model=model, teacher_model=model_teacher, keep_rate=EMA_KEEP_RATE)

        scalar_info = {
            'model_seg_total_loss': model_seg_total_loss.item(),
            'loss_target': loss_target.item(),
            'loss_mixed': loss_mixed.item(),
            'loss_pin': loss_pin.item(),
            'loss_pan': loss_pan.item(),

            'loss_recon': loss_recon.item(),
            'loss_sem': loss_sem.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_adv_target_all': loss_adv_target_all.item(),
            'loss_D_value_morph': loss_D_value_morph,
        }

        if args.tensorboard:
            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)
                writer.add_scalar('lr/lr_model', lr_model, i_iter)
                writer.add_scalar('lr/lr_morph', lr_morph, i_iter)
                writer.add_scalar('lr/lr_D_morph', lr_D_morph, i_iter)

                writer.add_scalar('miou/mIoU', mIoU, i_iter)
                writer.add_scalar('miou/mIoU_pin', mIoU_pin, i_iter)
                writer.add_scalar('miou/mIoU_pan', mIoU_pan, i_iter)
                writer.add_scalar('miou/mIoU_moe', mIoU_moe, i_iter)
                writer.add_scalar('miou/mIoU_teacher', mIoU_teacher, i_iter)

            if i_iter % SAVE_IMG_PRED_EVERY == 0:
                with torch.no_grad():
                    # draw
                    # 1. draw grid
                    logging.info('drawing visualization figures!')
                    grid_pin2pan = reg_model(grid_img.float(), flow_pin2pan)
                    grid_pin2pan_image = comput_fig(grid_pin2pan)
                    writer.add_figure('grid_pin2pan_image', grid_pin2pan_image, i_iter)
                    plt.close(grid_pin2pan_image)

                    grid_pan2pin = reg_model(grid_img.float(), flow_pan2pin)
                    grid_pan2pin_image = comput_fig(grid_pan2pin)
                    writer.add_figure('grid_pan2pin_image', grid_pan2pin_image, i_iter)
                    plt.close(grid_pan2pin_image)

                    grid_source_pin2pan = reg_model(grid_img.float(), flow_source_pin2pan)
                    grid_source_pin2pan_image = comput_fig(grid_source_pin2pan)
                    writer.add_figure('grid_source_pin2pan_image', grid_source_pin2pan_image, i_iter)
                    plt.close(grid_source_pin2pan_image)

                    grid_source_pan2pin = reg_model(grid_img.float(), flow_source_pan2pin)
                    grid_source_pan2pin_image = comput_fig(grid_source_pan2pin)
                    writer.add_figure('grid_source_pan2pin_image', grid_source_pan2pin_image, i_iter)
                    plt.close(grid_source_pan2pin_image)

                    # 2. draw image
                    # draw source pin
                    images_source_pin_origin_figure = comput_fig(images_source_pin_origin)
                    writer.add_figure('images_source_pin_origin_figure', images_source_pin_origin_figure, i_iter)
                    plt.close(images_source_pin_origin_figure)

                    images_source_pin_figure = comput_fig(images_source_pin)
                    writer.add_figure('images_source_pin_figure', images_source_pin_figure, i_iter)
                    plt.close(images_source_pin_figure)

                    images_pin2pan_figure = comput_fig(images_pin2pan)
                    writer.add_figure('images_pin2pan_figure', images_pin2pan_figure, i_iter)
                    plt.close(images_pin2pan_figure)

                    images_pin_with_grid = draw_with_grid(images_source_pin, grid_img)
                    images_pin2pan_with_grid = reg_model(images_pin_with_grid, flow_pin2pan)
                    images_pin2pan_with_grid_figure = comput_fig(images_pin2pan_with_grid)
                    writer.add_figure('images_pin2pan_with_grid_figure', images_pin2pan_with_grid_figure,
                                      i_iter)
                    plt.close(images_pin2pan_with_grid_figure)

                    images_pin_reconstruct_with_grid = reg_model(images_pin2pan_with_grid, flow_pan2pin)
                    images_pin_reconstruct_with_grid_figure = comput_fig(images_pin_reconstruct_with_grid)
                    writer.add_figure('images_pin_reconstruct_with_grid_figure',
                                      images_pin_reconstruct_with_grid_figure, i_iter)
                    plt.close(images_pin_reconstruct_with_grid_figure)

                    images_source_pin2pan_figure = comput_fig(images_source_pin2pan)
                    writer.add_figure('images_source_pin2pan_figure', images_source_pin2pan_figure, i_iter)
                    plt.close(images_source_pin2pan_figure)

                    images_pin_with_grid = draw_with_grid(images_source_pin, grid_img)
                    images_source_pin2pan_with_grid = reg_model(images_pin_with_grid, flow_source_pin2pan)
                    images_source_pin2pan_with_grid_figure = comput_fig(images_source_pin2pan_with_grid)
                    writer.add_figure('images_source_pin2pan_with_grid_figure', images_source_pin2pan_with_grid_figure,
                                      i_iter)
                    plt.close(images_source_pin2pan_with_grid_figure)

                    images_source_pin_reconstruct_with_grid = reg_model(images_source_pin2pan_with_grid,
                                                                        flow_source_pan2pin)
                    images_source_pin_reconstruct_with_grid_figure = comput_fig(images_source_pin_reconstruct_with_grid)
                    writer.add_figure('images_source_pin_reconstruct_with_grid_figure',
                                      images_source_pin_reconstruct_with_grid_figure, i_iter)
                    plt.close(images_source_pin_reconstruct_with_grid_figure)

                    # draw source pan
                    images_target_cropped_figure = comput_fig(images_target_cropped)
                    writer.add_figure('images_target_cropped_figure', images_target_cropped_figure, i_iter)
                    plt.close(images_target_cropped_figure)

                    images_pan_with_grid = draw_with_grid(images_target_cropped, grid_img)
                    images_pan2pin_with_grid = reg_model(images_pan_with_grid, flow_pan2pin)
                    images_pan2pin_with_grid_figure = comput_fig(images_pan2pin_with_grid)
                    writer.add_figure('images_pan2pin_with_grid_figure', images_pan2pin_with_grid_figure,
                                      i_iter)
                    plt.close(images_pan2pin_with_grid_figure)

                    images_source_pan_origin_figure = comput_fig(images_source_pan_origin)
                    writer.add_figure('images_source_pan_origin_figure', images_source_pan_origin_figure, i_iter)
                    plt.close(images_source_pan_origin_figure)

                    images_source_pan_figure = comput_fig(images_source_pan)
                    writer.add_figure('images_source_pan_figure', images_source_pan_figure, i_iter)
                    plt.close(images_source_pan_figure)

                    images_source_pan_with_grid = draw_with_grid(images_source_pan, grid_img)
                    images_source_pan2pin_with_grid = reg_model(images_source_pan_with_grid, flow_source_pan2pin)
                    images_source_pan2pin_with_grid_figure = comput_fig(images_source_pan2pin_with_grid)
                    writer.add_figure('images_source_pan2pin_with_grid_figure', images_source_pan2pin_with_grid_figure,
                                      i_iter)
                    plt.close(images_source_pan2pin_with_grid_figure)

                    # [option] closer look at target labels
                    labels_target_cropped_pseudo_figure = comput_fig(get_colored_labels(labels_target_cropped_ensemble))
                    writer.add_figure('labels_target_cropped_pseudo_figure', labels_target_cropped_pseudo_figure,
                                      i_iter)
                    plt.close(labels_target_cropped_pseudo_figure)

                    images_target_figure = comput_fig(images_target)
                    writer.add_figure('images_target_figure', images_target_figure, i_iter)
                    plt.close(images_target_figure)

                    images_target_strong_figure = comput_fig(images_target_strong)
                    writer.add_figure('images_target_strong_figure', images_target_strong_figure, i_iter)
                    plt.close(images_target_strong_figure)

                    labels_target_pseudo_figure = comput_fig(get_colored_labels(labels_target_ensemble))
                    writer.add_figure('labels_target_pseudo_figure', labels_target_pseudo_figure,
                                      i_iter)
                    plt.close(labels_target_pseudo_figure)

                    # mixed pin2pan&target
                    mixed_images_pin2pan_figure = comput_fig(mixed_images_pin2pan)
                    writer.add_figure('mixed_images_pin2pan_figure', mixed_images_pin2pan_figure, i_iter)
                    plt.close(mixed_images_pin2pan_figure)
                    # mixed pan&target
                    mixed_images_pan_figure = comput_fig(mixed_images_pan)
                    writer.add_figure('mixed_images_pan_figure', mixed_images_pan_figure, i_iter)
                    plt.close(mixed_images_pan_figure)
                    # pan&pin2pan
                    mixed_images_source_pin2pan_figure = comput_fig(mixed_images_source_pin2pan)
                    writer.add_figure('mixed_images_source_pin2pan_figure', mixed_images_source_pin2pan_figure, i_iter)
                    plt.close(mixed_images_source_pin2pan_figure)

                    # grey figure for pixel dis；black(0) means pin-like；white(1) means pan-like
                    D_out_full_pin2pan_figure = comput_fig(torch.sigmoid(D_out_full_pin2pan))
                    writer.add_figure('D_out_full_pin2pan_figure', D_out_full_pin2pan_figure, i_iter)
                    plt.close(D_out_full_pin2pan_figure)

                    D_out_full_source_pin2pan_figure = comput_fig(torch.sigmoid(D_out_full_source_pin2pan))
                    writer.add_figure('D_out_full_source_pin2pan_figure', D_out_full_source_pin2pan_figure, i_iter)
                    plt.close(D_out_full_source_pin2pan_figure)

                    draw_option = False
                    if draw_option:
                        # background ignore is white
                        # [option] closer look at source labels
                        labels_pin_figure = comput_fig(get_colored_labels(labels_source_pin))
                        writer.add_figure('labels_pin_figure', labels_pin_figure, i_iter)
                        plt.close(labels_pin_figure)
                        # [option] closer look at source2target labels
                        labels_pin2pan_figure = comput_fig(get_colored_labels(labels_pin2pan))
                        writer.add_figure('labels_pin2pan_figure', labels_pin2pan_figure, i_iter)
                        plt.close(labels_pin2pan_figure)
                        # [option] closer look at mixed pin2pan&target labels
                        mixed_labels_pin2pan_figure = comput_fig(get_colored_labels(mixed_labels_pin2pan))
                        writer.add_figure('mixed_labels_pin2pan_figure', mixed_labels_pin2pan_figure, i_iter)
                        plt.close(mixed_labels_pin2pan_figure)
                        # [option] closer look at mixed pan&target labels
                        mixed_labels_pan_figure = comput_fig(get_colored_labels(mixed_labels_pan))
                        writer.add_figure('mixed_labels_pan_figure', mixed_labels_pan_figure, i_iter)
                        plt.close(mixed_labels_pan_figure)

                    logging.info('draw visualization figures done!')

        if i_iter % 10 == 0:
            logging.info('iter = {0:8d}/{1:8d}, losses:{2}'.format(
                i_iter, args.num_steps, scalar_info))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            logging.info('taking snapshot ...')
            freeze_model(model)

            eval_result = evaluator(model_teacher, i_iter)
            freeze_model(model_teacher)
            mIoU_teacher = eval_result['mIoU_ensemble']

            eval_result_all = evaluator(model, i_iter)
            mIoU = eval_result_all['mIoU_ensemble']
            mIoU_pin = eval_result_all['mIoUs'][0]
            mIoU_pan = eval_result_all['mIoUs'][1]
            mIoU_moe = eval_result_all['mIoUs'][2]
            best_miou_str = eval_result_all['best_miou_str']

            if mIoU >= bestIoU:
                bestIoU = mIoU
                pre_filename = osp.join(args.snapshot_dir + 'best*.pth')
                pre_filename = glob.glob(pre_filename)
                try:
                    for p in pre_filename:
                        os.remove(p)
                except OSError as e:
                    logging.info(e)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                torch.save(model_teacher.state_dict(),
                           osp.join(args.snapshot_dir, 'best_teacher.pth'))
                torch.save(model_morph.state_dict(),
                           osp.join(args.snapshot_dir, 'best_morph.pth'))
                torch.save(model_D_morph_pin2pan.state_dict(),
                           osp.join(args.snapshot_dir, 'best_D_morph_pin2pan.pth'))
                torch.save(model_D_morph_source_pin2pan.state_dict(),
                           osp.join(args.snapshot_dir, 'best_D_morph_source_pin2pan.pth'))
                with open(osp.join(args.snapshot_dir, f'{TIME_STAMP}_best_miou.txt'), mode='w', encoding='utf-8') as f:
                    f.write(best_miou_str)
            unfreeze_model(model)

        if i_iter >= args.num_steps_stop:
            logging.info('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, 'latest.pth'))
            torch.save(model_teacher.state_dict(),
                       osp.join(args.snapshot_dir, 'latest_teacher.pth'))
            torch.save(model_morph.state_dict(),
                       osp.join(args.snapshot_dir, 'latest_morph.pth'))
            torch.save(model_D_morph_pin2pan.state_dict(),
                       osp.join(args.snapshot_dir, 'latest_D_morph_pin2pan.pth'))
            torch.save(model_D_morph_source_pin2pan.state_dict(),
                       osp.join(args.snapshot_dir, 'latest_D_morph_source_pin2pan.pth'))
            break

        if i_iter != 0 and i_iter % SAVE_CKPT_EVERY == 0:
            logging.info('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}.pth'))
            torch.save(model_teacher.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}_teacher.pth'))
            torch.save(model_morph.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}_morph.pth'))
            torch.save(model_D_morph_pin2pan.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}_D_morph_pin2pan.pth'))
            torch.save(model_D_morph_source_pin2pan.state_dict(),
                       osp.join(args.snapshot_dir, f'iter{i_iter}_D_morph_source_pin2pan.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
