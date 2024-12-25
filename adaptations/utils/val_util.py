import numpy as np
import torch
import torch.nn.functional as F

from adaptations.compute_iou import fast_hist, per_class_iu


@torch.no_grad()
class Evaluator:

    def __init__(self, testloader, num_classes, NAME_CLASSES, device, softmax=False, aux_fusion=False):
        self.testloader = testloader
        self.num_classes = num_classes
        self.NAME_CLASSES = NAME_CLASSES
        self.device = device
        self.softmax = softmax
        self.aux_fusion = aux_fusion

    @torch.no_grad()
    def __call__(self, models, i_iter=-1, rates=None):
        if type(models) == list:
            num_models = len(models)
        else:
            models = [models]
            num_models = 1
        if rates is None:
            rates = [1.0]
        assert num_models == len(rates)

        for i in range(num_models):
            models[i].eval()

        hist_ensemble = np.zeros((self.num_classes, self.num_classes))
        hists = [np.zeros((self.num_classes, self.num_classes)) for _ in range(num_models)]
        for index, batch in enumerate(self.testloader):
            if index % 10 == 0:
                print(f'evaluating {index}/{len(self.testloader)}')
            image, label, _, name = batch
            label = label.cpu().data[0].numpy()

            with torch.no_grad():
                input_image = image.to(self.device)
                output_list = []
                for i in range(num_models):
                    if self.aux_fusion:
                        outputs, output = models[i](input_image)
                        output_temp = sum(outputs) + output
                    else:
                        _, output_temp = models[i](input_image)

                    if self.softmax:
                        output_temp = F.softmax(output_temp, dim=1)
                    output_list.append(output_temp)

            output_ensemble = torch.zeros_like(output_list[0])
            for i in range(num_models):
                output_ensemble += output_list[i] * rates[i]
            output_ensemble /= num_models

            output_ensemble = output_ensemble.cpu().data[0].numpy()
            output_ensemble = output_ensemble.transpose(1, 2, 0)
            output_ensemble = np.asarray(np.argmax(output_ensemble, axis=2), dtype=np.uint8)
            hist_ensemble += fast_hist(label.flatten(), output_ensemble.flatten(), self.num_classes)

            for i in range(num_models):
                output = output_list[i].cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                hists[i] += fast_hist(label.flatten(), output.flatten(), self.num_classes)

        best_miou_str = '\n' + '-' * 10 + '\n'

        mIoUs = []
        for i in range(num_models):
            mIoU = per_class_iu(hists[i])
            for ind_class in range(self.num_classes):
                temp_str = '===>{:<15}:\t{}'.format(self.NAME_CLASSES[ind_class],
                                                    str(round(mIoU[ind_class] * 100, 2)))
                print(temp_str)
                best_miou_str += f'{temp_str}\n'
            mIoU = round(np.nanmean(mIoU) * 100, 2)
            mIoUs.append(mIoU)
            print(f'===> mIoU-[{i + 1}]: ' + str(mIoU))
            print('-' * 10)
            best_miou_str += f'best mIoU_[{i}] = {mIoU}, best iter = {i_iter}\n'
            best_miou_str += '\n' + '-' * 10 + '\n'

        mIoU_ensemble = per_class_iu(hist_ensemble)
        for ind_class in range(self.num_classes):
            temp_str = '===>{:<15}:\t{}'.format(self.NAME_CLASSES[ind_class],
                                                str(round(mIoU_ensemble[ind_class] * 100, 2)))
            print(temp_str)
            best_miou_str += f'{temp_str}\n'
        mIoU_ensemble = round(np.nanmean(mIoU_ensemble) * 100, 2)
        print('===> mIoU_ensemble: ' + str(mIoU_ensemble))
        best_miou_str += f'best mIoU_ensemble = {mIoU_ensemble}, best iter = {i_iter}\n'

        for i in range(num_models):
            models[i].train()

        return {
            'mIoU_ensemble': mIoU_ensemble,
            'mIoUs': mIoUs,
            'best_miou_str': best_miou_str
        }


@torch.no_grad()
class MultiHeadEvaluator:

    def __init__(self, testloader, num_classes, NAME_CLASSES, device, softmax=False, aux_fusion=False):
        self.testloader = testloader
        self.num_classes = num_classes
        self.NAME_CLASSES = NAME_CLASSES
        self.device = device
        self.softmax = softmax
        self.aux_fusion = aux_fusion

    @torch.no_grad()
    def __call__(self, model, i_iter=-1):
        NUM_HEADS = 3
        model.eval()

        hist_ensemble = np.zeros((self.num_classes, self.num_classes))
        hists = [np.zeros((self.num_classes, self.num_classes)) for _ in range(NUM_HEADS)]
        for index, batch in enumerate(self.testloader):
            if index % 10 == 0:
                print(f'evaluating {index}/{len(self.testloader)}')
            image, label, _, name = batch
            label = label.cpu().data[0].numpy()

            with torch.no_grad():
                input_image = image.to(self.device)

                outputs, output = model(input_image)
                output_temp = outputs + [output]

                if self.softmax:
                    output_list = [F.softmax(x, dim=1) for x in output_temp]
                else:
                    output_list = output_temp

            output_ensemble = sum(output_list) / len(output_list)

            output_ensemble = output_ensemble.cpu().data[0].numpy()
            output_ensemble = output_ensemble.transpose(1, 2, 0)
            output_ensemble = np.asarray(np.argmax(output_ensemble, axis=2), dtype=np.uint8)
            hist_ensemble += fast_hist(label.flatten(), output_ensemble.flatten(), self.num_classes)

            for i in range(len(output_list)):
                output = output_list[i].cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                hists[i] += fast_hist(label.flatten(), output.flatten(), self.num_classes)

        best_miou_str = '\n' + '-' * 10 + '\n'

        mIoUs = []
        for i in range(len(output_list)):
            mIoU = per_class_iu(hists[i])
            for ind_class in range(self.num_classes):
                temp_str = '===>{:<15}:\t{}'.format(self.NAME_CLASSES[ind_class],
                                                    str(round(mIoU[ind_class] * 100, 2)))
                print(temp_str)
                best_miou_str += f'{temp_str}\n'
            mIoU = round(np.nanmean(mIoU) * 100, 2)
            mIoUs.append(mIoU)
            print(f'===> mIoU-[{i + 1}]: ' + str(mIoU))
            print('-' * 10)
            best_miou_str += f'best mIoU_[{i}] = {mIoU}, best iter = {i_iter}\n'
            best_miou_str += '\n' + '-' * 10 + '\n'

        mIoU_ensemble = per_class_iu(hist_ensemble)
        for ind_class in range(self.num_classes):
            temp_str = '===>{:<15}:\t{}'.format(self.NAME_CLASSES[ind_class],
                                                str(round(mIoU_ensemble[ind_class] * 100, 2)))
            print(temp_str)
            best_miou_str += f'{temp_str}\n'
        mIoU_ensemble = round(np.nanmean(mIoU_ensemble) * 100, 2)
        print('===> mIoU_ensemble: ' + str(mIoU_ensemble))
        best_miou_str += f'best mIoU_ensemble = {mIoU_ensemble}, best iter = {i_iter}\n'

        model.train()

        return {
            'mIoU_ensemble': mIoU_ensemble,
            'mIoUs': mIoUs,
            'best_miou_str': best_miou_str
        }
