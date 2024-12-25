import numpy as np
import torch


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def oneMix(mask, data=None, target=None):
    # Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target


def mix_source_target(images_source, labels_source, images_target, labels_target, labels_target_weight,
                      ignore_label=None, need_class_mask=False):
    batch_size = images_source.shape[0]
    mixed_images = []
    mixed_labels = []
    mixed_weights = []
    labels_source_weight = torch.ones_like(labels_target_weight).to(labels_target_weight.device)
    if need_class_mask:
        class_masks = []
    for image_i in range(batch_size):
        classes = torch.unique(labels_source[image_i])
        if ignore_label is not None:
            classes_new = []
            for idx in classes:
                if idx != ignore_label:
                    classes_new.append(idx)
            classes = torch.tensor(classes_new, dtype=torch.long)

        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(
            np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).to(
            labels_source.device)

        class_mask = generate_class_mask(labels_source[image_i], classes).unsqueeze(0)

        mixed_image, mixed_label = oneMix(mask=class_mask,
                                          data=torch.stack([images_source[image_i],
                                                            images_target[image_i]], dim=0),
                                          target=torch.stack([labels_source[image_i],
                                                              labels_target[image_i]], dim=0))
        mixed_weight, _ = oneMix(mask=class_mask,
                                 data=torch.stack([labels_source_weight[image_i],
                                                   labels_target_weight[image_i]], dim=0))
        mixed_images.append(mixed_image)
        mixed_labels.append(mixed_label)
        mixed_weights.append(mixed_weight)
        if need_class_mask:
            class_masks.append(class_mask)
    if need_class_mask:
        return torch.cat(mixed_images, dim=0), torch.cat(mixed_labels, dim=0), torch.cat(mixed_weights, dim=0), \
            torch.cat(class_masks, dim=0)
    else:
        return torch.cat(mixed_images, dim=0), torch.cat(mixed_labels, dim=0), torch.cat(mixed_weights, dim=0)
