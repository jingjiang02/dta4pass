# this code heavily based on detectron2

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import filters


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_identity_grid(H, W):
    # 生成标准化坐标网格
    xx = torch.linspace(-1, 1, H)
    yy = torch.linspace(-1, 1, W)
    grid_y, grid_x = torch.meshgrid(yy, xx)
    grid = torch.stack((grid_y, grid_x), -1)
    grid = grid.unsqueeze(0)

    return grid


def create_dual_cylindrical_grid(H, W, scale=1.0):
    # 生成标准化坐标网格
    xx = torch.linspace(-1, 1, H)
    yy = torch.linspace(-1, 1, W)
    grid_y, grid_x = torch.meshgrid(yy, xx)

    # 水平方向上的双柱面坐标
    half_width = H // 2
    theta_x_left = torch.atan(grid_x[:, :half_width]) * 2
    theta_x_right = torch.atan(grid_x[:, half_width:]) * 2
    x_cyl_left = torch.sin(theta_x_left * scale)
    x_cyl_right = torch.sin(theta_x_right * scale)
    x_cyl = torch.cat((x_cyl_left, x_cyl_right), dim=1)

    # 垂直方向保持常规柱面变换
    y_cyl = grid_y

    # 创建网格
    grid = torch.stack((y_cyl, x_cyl), 2)
    grid = grid.unsqueeze(0)
    return grid


def create_cylindrical_grid(H, W, focal_length):
    # 生成像素坐标网格
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
    ys, xs = ys.to(torch.float32), xs.to(torch.float32)

    # 转换到圆柱坐标
    theta = xs * W / (2 * focal_length)
    xs = torch.sin(theta)
    ys = ys * W / (2 * focal_length)
    zs = torch.cos(theta)

    # 归一化坐标
    cylindrical_coords = torch.stack([xs, ys, zs], -1)
    cylindrical_coords = cylindrical_coords / cylindrical_coords.norm(dim=-1, keepdim=True)

    # 转换回图像坐标系
    grid = cylindrical_coords[..., :2] / cylindrical_coords[..., 2:]
    grid = grid.permute(1, 0, 2)  # 调整维度以符合grid_sample的要求

    return grid.unsqueeze(0)  # 增加一个批处理维度


def comput_fig(img, rgb_range=1, total_num=4):
    sqart_total_num = int(total_num ** 0.5)
    img = img.detach().cpu().numpy()[0:total_num, ...]
    fig = plt.figure(figsize=(12, 6), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(sqart_total_num, sqart_total_num, i + 1)
        plt.axis('off')
        img_local = img[i]

        img_local = np.transpose(img_local, (1, 2, 0))
        if img_local.shape[-1] == 1:
            # gray
            plt.imshow(img_local, cmap='gray')
        else:
            if rgb_range != 1:
                # convert 0-255 to long
                img_local = img_local.astype(np.uint8)
            plt.imshow(img_local)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j:j + line_thickness, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i:i + line_thickness] = 1
    grid_img = grid_img[:, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


class UnNormalize(torch.nn.Module):
    # restore from T.Normalize
    # 反归一化
    def __init__(self, device, mean=(.485, .456, .406), std=(.229, .224, .225)):
        super().__init__()
        self.mean = torch.tensor(mean).view((1, -1, 1, 1)).to(device)
        self.std = torch.tensor(std).view((1, -1, 1, 1)).to(device)

    def __call__(self, x):
        x = (x * self.std) + self.mean
        return torch.clip(x, 0.0, 1.0)


def draw_with_grid(image, mask):
    def_out_with_grid = image.clone()
    def_out_with_grid[:, 0:1, :, :][mask == 1] = 1  # red
    def_out_with_grid[:, 1:2, :, :][mask == 1] = 0
    def_out_with_grid[:, 2:3, :, :][mask == 1] = 0
    return def_out_with_grid


# 准备计算梯度
# 定义Sobel滤波器的水平和垂直方向的卷积核
G_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
G_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
G_x = G_x.cuda()
G_y = G_y.cuda()
gray_weights = gray_weights.cuda()
G_x.requires_grad = False
G_y.requires_grad = False
gray_weights.requires_grad = False


def get_grad(img, convert2gray=False):
    # 将图像转换为灰度
    # 使用加权平均来模拟人眼对不同颜色的敏感度
    if convert2gray:
        gray_img = torch.sum(img * gray_weights, dim=1, keepdim=True)
    else:
        gray_img = img
    # 使用 Sobel 算子计算水平和垂直梯度
    grad_x = F.conv2d(gray_img, G_x, padding=1)
    grad_y = F.conv2d(gray_img, G_y, padding=1)

    return torch.cat([img, grad_x, grad_y], dim=1)


def warp_labels(labels, flow, reg_model, temp_value=100, ignore_label=255):
    labels = labels.detach().clone()
    labels[labels == 0] = temp_value
    labels_warped = reg_model(labels[:, None, :, :].float(), flow).long()
    labels_warped = labels_warped.squeeze(1)
    labels_warped[labels_warped == 0] = ignore_label
    labels_warped[labels_warped == temp_value] = 0
    # labels_source back
    labels[labels == temp_value] = 0
    return labels_warped


def get_confidence(pred, labels, fill_value=1.0, ignore_label=255, sample_rate=0.5, exp_val=False, squeeze=True):
    pred_softmax = F.softmax(pred, dim=1)
    labels_temp = labels.detach().clone()
    ignore_mask = labels_temp == ignore_label
    labels_temp[ignore_mask] = 0
    confidence = pred_softmax.gather(1, labels_temp.unsqueeze(1))
    confidence = confidence.squeeze(1)
    confidence[ignore_mask] = fill_value
    confidence = F.interpolate(confidence[:, None, :, :], scale_factor=sample_rate, mode='bilinear').detach()
    if squeeze:
        confidence = confidence.squeeze(1)
    if exp_val:
        confidence = torch.exp(confidence)
        min_val = confidence.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = confidence.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        confidence = (confidence - min_val) / (max_val - min_val + 1e-8)
    return confidence


def get_confidence_by_label(labels, num_classes, bound_value=1.0, no_bound_value=0.75, sample_rate=0.5):
    labels_temp = labels.detach().clone().cpu().numpy()
    bound = np.zeros_like(labels_temp)
    for b in range(labels_temp.shape[0]):
        for c in range(num_classes):
            class_mask = (labels_temp[b] == c)
            boundary = filters.sobel(class_mask)
            bound[b] |= (boundary > 0)
    bound = torch.tensor(bound, dtype=torch.float32).to(labels.device)
    bound[bound == 0.0] = no_bound_value
    bound[bound == 1.0] = bound_value
    confidence = F.interpolate(bound[:, None, :, :].float(), scale_factor=sample_rate, mode='bilinear').squeeze(
        1).detach()
    return confidence


palette = [
    [128, 64, 128],  # 1 Road
    [244, 35, 232],  # 2 S.walk
    [70, 70, 70],  # 3 Build
    [102, 102, 156],  # 4 Wall
    [190, 153, 153],  # 5 Fence
    [153, 153, 153],  # 6 Pole
    [250, 170, 30],  # 7 Tr. light
    [220, 220, 0],  # 8 Tr. sign
    [107, 142, 35],  # 9 Veget
    [152, 251, 152],  # 10 Terrain
    [70, 130, 180],  # 11 Sky
    [220, 20, 60],  # 12 Person
    [0, 0, 142],  # 13 Car
]


@torch.no_grad()
def get_colored_labels(labels, ignore_label=255):
    # labels.shape = [B, H, W]
    num_classes = len(palette)
    labels_temp = labels.detach().clone()[:, None, :, :].repeat(1, 3, 1, 1)
    # labels_temp.shape = [B, 3, H, W]
    for l in range(num_classes):
        for c in range(3):
            labels_temp_c = labels_temp[:, c:c + 1, :, :]
            labels_temp_c[labels_temp_c == l] = palette[l][c]
    labels_temp[labels_temp == ignore_label] = 0
    return labels_temp
