import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def gen_pseudo_label(model, dataloader, save_dir, num_class=13, multi_scale=False):
    # set to eval mode
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    interp = nn.Upsample(size=(400, 2048), mode='bilinear', align_corners=True)

    # generate pseudo labels for dp train set
    print('generate pseudo labels for dp train set...')
    predicted_label = np.zeros((len(dataloader), 400, 2048), dtype=np.int8)
    predicted_prob = np.zeros((len(dataloader), 400, 2048), dtype=np.float16)
    image_name = []

    for index, batch in enumerate(dataloader):
        if index % 10 == 0:
            print('{}/{} processed'.format(index, len(dataloader)))

        image, _, name = batch
        image_name.append(name[0])
        image = image.cuda()
        b, c, h, w = image.shape
        output_temp = torch.zeros((b, num_class, h, w), dtype=image.dtype).cuda()
        if multi_scale:
            # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # ms
            # scales = [0.75, 1.0, 1.75]  # ms
            raise Exception
        else:
            scales = [1]  # origin no scale
        for sc in scales:
            new_h, new_w = int(sc * h), int(sc * w)
            img_tem = nn.UpsamplingBilinear2d(size=(new_h, new_w))(image)
            with torch.no_grad():
                _, output = model(img_tem)
                output_temp += interp(output)
        output = output_temp / len(scales)
        output = F.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)

        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label
        predicted_prob[index] = prob
    thres = []
    for i in range(num_class):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        thres.append(np.median(x))
    print(thres)
    thres = np.array(thres)
    thres[thres > 0.9] = 0.9
    print(thres)
    for index in range(len(dataloader)):
        name = image_name[index]
        label = predicted_label[index]
        label = np.asarray(label, dtype=np.uint8)
        prob = predicted_prob[index]
        for i in range(num_class):
            label[(prob < thres[i]) * (label == i)] = 255
        output = label
        output = Image.fromarray(output)
        name = name.replace('.jpg', '_labelTrainIds.png')
        save_fn = os.path.join(save_dir, name)
        if not os.path.exists(os.path.dirname(save_fn)):
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        output.save(save_fn)
    print('pseudo label generated!')


@torch.no_grad()
def get_pseudo_label_online(pred, ignore_label, device, n_rate=0.5, threshold=0.9, need_weight=False, softmax=True):
    B, num_class, H, W = pred.shape
    pseudo_label_list = []
    if need_weight:
        pseudo_label_weight_list = []
    for i in range(B):
        output = pred[i]
        if softmax:
            output = F.softmax(output, dim=0)
        output = output.detach().cpu().numpy()
        label, prob = np.argmax(output, axis=0), np.max(output, axis=0)

        predicted_label = label.copy()
        predicted_prob = prob.copy()

        thres = []
        for i in range(num_class):
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int32(np.round(len(x) * n_rate))])
        thres = np.array(thres)
        thres[thres > threshold] = threshold

        for i in range(num_class):
            label[(prob < thres[i]) * (label == i)] = ignore_label

        if need_weight:
            ps_large_p = torch.tensor(label != ignore_label)
            ps_size = np.size(label)
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                predicted_prob.shape)
            pseudo_label_weight_list.append(pseudo_weight)

        pseudo_label = torch.LongTensor(np.array(label).astype('int32'))
        pseudo_label_list.append(pseudo_label)
    pseudo_label_list = torch.stack(pseudo_label_list, dim=0).to(device)

    if need_weight:
        pseudo_label_weight_list = torch.stack(pseudo_label_weight_list, dim=0).to(device)
        return pseudo_label_list, pseudo_label_weight_list
    else:
        return pseudo_label_list


class VarianceLoss:
    def __init__(self, ignore_index=255):
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.kl_distance = nn.KLDivLoss(reduction='none')
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)

    def __call__(self, labels, pred1, pred2):
        loss = self.criterion(pred1, labels)

        variance = torch.sum(self.kl_distance(self.log_sm(pred1), self.sm(pred2)), dim=1)
        exp_variance = torch.exp(-variance)
        loss = torch.mean(loss * exp_variance) + torch.mean(variance)
        return loss
