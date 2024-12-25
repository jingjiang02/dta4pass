from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-5


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)


def kd_loss(feats, select_feat_self):
    T = 100
    alpha = 0.9
    loss_kl_self = nn.KLDivLoss()(
        F.log_softmax(feats / T, dim=1),
        F.softmax(select_feat_self / T, dim=1)) * (alpha * T * T) + F.cross_entropy(feats,
                                                                                    torch.argmax(select_feat_self,
                                                                                                 dim=1).long()) * (
                           1. - alpha)
    return loss_kl_self


def feat_kl_loss(feats, labels, feats_mem):
    B, C, H, W = feats.shape

    _, H_org, W_org = labels.shape
    labels = F.interpolate(labels.unsqueeze(1).float(), (H, W), mode='nearest')

    select_feat = torch.clone(feats)
    feats = feats.permute(0, 2, 3, 1).contiguous().view(-1, C)
    select_feat = select_feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    feats_mem = feats_mem.squeeze(1)
    batch_feats_mem = torch.zeros_like(feats_mem)

    ignore_index = 255
    for c in labels.unique():
        if c == ignore_index: continue
        c = c.item()
        feats_cls = feats[labels == c].mean(0)
        batch_feats_mem[int(c)] = feats_cls

        m = labels == c
        m = m[..., None].repeat(1, C)
        feat_temp = feats_mem[int(c)][None, ...].expand(labels.shape[0], -1)
        select_feat = torch.where(m, feat_temp, select_feat)
    feats = feats.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    select_feat = select_feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    T = 20
    alpha = 0.9
    loss_kl = nn.KLDivLoss()(
        F.log_softmax(feats / T, dim=1),
        F.softmax(select_feat / T, dim=1)) * (alpha * T * T) + F.cross_entropy(feats, torch.argmax(select_feat,
                                                                                                   dim=1).long()) * (
                      1. - alpha)

    return loss_kl, batch_feats_mem, select_feat


def co_kl_loss(pred_source_list, pred_source_others_list):
    assert len(pred_source_list) == len(pred_source_others_list)
    num = len(pred_source_list)
    loss_kl_all = 0.
    for i in range(num):
        loss_kl = nn.KLDivLoss()(F.log_softmax(pred_source_list[i], dim=1),
                                 F.softmax(pred_source_others_list[i], dim=1))
        loss_kl_all += loss_kl
    loss_kl_all /= num
    return loss_kl_all


class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        assert loss.shape == pixelWiseWeight.shape  # check shape
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


class BCEWithLogitsLossPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(BCEWithLogitsLossPixelWiseWeighted, self).__init__()
        self.BCE = torch.nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.BCE(output, target)
        assert loss.shape == pixelWiseWeight.shape  # check shape
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


class PCLLoss:
    def __init__(self, reg_weight=0, tau=0.1, proto_dims=128, num_samples=4096, ignore_index=255):
        if num_samples == -1:
            num_samples = 32768  # 128*256
        self.num_samples = num_samples
        self.proto_dims = proto_dims
        self.ignore_index = ignore_index
        self.tau = tau
        self.reg_weight = reg_weight

    def _process(self, v, label):
        b, d, h, w = v.shape

        label = label[:, None, :, :]

        label = F.interpolate(
            input=label.float(),
            size=(h, w),
            mode='nearest'
        ).long()
        v = v.permute(0, 2, 3, 1).contiguous().view(-1, self.proto_dims)
        label = label.view(-1)

        not_ignore_idx = torch.where(label < self.ignore_index)
        v = v[not_ignore_idx]
        label = label[not_ignore_idx]
        v = F.normalize(v, p=2, dim=1)

        return v, label

    def __call__(self, v, label, prototypes):
        v, label = self._process(v, label)
        contrastive_loss = self.contrastive_loss(v, label, prototypes)
        reg_loss = self.reg_loss(v, prototypes)
        return contrastive_loss + reg_loss

    def contrastive_loss(self, v, label, prototypes):
        indices = torch.randperm(v.shape[0])[:self.num_samples]
        v_samples = v[indices]
        lbl_samples = label[indices]

        loss = nn.CrossEntropyLoss()((v_samples @ prototypes.T) / self.tau, lbl_samples)
        return loss

    def reg_loss(self, v, prototypes):
        v_mean = v.mean(axis=0, keepdim=True)

        logits = v_mean.mm(prototypes.detach().permute(1, 0)) / self.tau
        loss = torch.sum(torch.softmax(logits, dim=1).log()) * self.reg_weight

        return loss


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, temperature=0.1, base_temperature=0.07, ignore_label=255, max_samples=1024, max_views=100):
        super(PixelContrastLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_label = ignore_label
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()

        predict = predict.detach().clone()
        predict = torch.argmax(predict, dim=1, keepdim=True).float()
        predict = torch.nn.functional.interpolate(predict,
                                                  (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss


class LogitConstraintLoss(nn.CrossEntropyLoss):
    """
    CrossEntropyLoss after Logit Norm.
    """

    def __init__(self,
                 weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 eps=1e-7):
        super(LogitConstraintLoss, self).__init__(weight,
                                                  size_average,
                                                  ignore_index,
                                                  reduce,
                                                  reduction,
                                                  label_smoothing)
        self.eps = eps

    def forward(self, cls_score, label):
        """Forward function."""
        norms = torch.norm(cls_score, p=2, dim=1, keepdim=True) + self.eps
        normed_logit = torch.div(cls_score, norms)
        loss_cls = super(LogitConstraintLoss, self).forward(normed_logit,
                                                            label)
        return loss_cls


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def __call__(self, loss, pred_a, pred_b):
        variance = torch.sum(self.kl_distance(self.log_sm(pred_a), self.sm(pred_b)), dim=1)
        exp_variance = torch.exp(-variance)
        assert loss.shape == exp_variance.shape  # check shape
        loss = torch.mean(loss * exp_variance) + torch.mean(variance)
        return loss


class UncertaintyDualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def __call__(self, loss, pred_a, pred_b):
        variance = (self.kl_distance(self.log_sm(pred_a), self.sm(pred_b)) +
                    self.kl_distance(self.log_sm(pred_b), self.sm(pred_a))) / 2.0
        variance = torch.sum(variance, dim=1)
        exp_variance = torch.exp(-variance)
        assert loss.shape == exp_variance.shape  # check shape
        loss = torch.mean(loss * exp_variance) + torch.mean(variance)
        return loss
