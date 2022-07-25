import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_perimage(pred_all, gt_all):
    dice_list = []
    hd95_list = []
    iou_list = []
    for i in range(pred_all.shape[0]):
        pred = pred_all[i]
        gt = gt_all[i]
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum()>0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            intersection = np.logical_and(pred, gt)
            union = np.logical_or(pred, gt)
            iou = np.sum(intersection) / np.sum(union)
            dice_list.append(dice)
            iou_list.append(iou)
            hd95_list.append(hd95)
        elif pred.sum() > 0 and gt.sum()==0:
            dice_list.append(1)
            #iou_list.append(0)
            hd95_list.append(0)
        else:
            dice_list.append(1)
            iou_list.append(1)
            hd95_list.append(0)
    return np.mean(dice_list), np.mean(hd95_list), np.mean(iou_list)




def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        iou = np.sum(intersection) / np.sum(union)
        return dice, hd95, iou
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1
    else:
        return 1, 0, 1

def sample_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_input = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
            net.train()
            with torch.no_grad():
                outputs, distributions = net(input, None)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.train()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        for level in range(len(prediction)):
            np.save(os.path.join(test_save_path, f"level{level}_" + case + "_prediction.npy"), np.rot90(prediction.astype(np.float32)[level].squeeze(), k=3))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_input = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs, distributions = net(input, None)
                print(torch.mean(torch.concat([d.loc.flatten() for d in distributions])))
                print(torch.mean(torch.concat([d.scale.flatten() for d in distributions])))
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list



def sample_single_image(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_input = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
            net.train()
            with torch.no_grad():
                outputs, distributions = net(input, None)
                #print(torch.mean(torch.concat([d.loc.flatten() for d in distributions])))
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.train()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_perimage(prediction == i, label == i))

    if test_save_path is not None:
        case = case.replace("/", "@")
        np.save(os.path.join(test_save_path, f"{case}_prediction.npy"), prediction.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_label.npy"), label.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_image.npy"), image.astype(np.float32))
    return metric_list

def test_single_image(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_input = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs, distributions = net(input, label_input)
                #print(torch.mean(torch.concat([d.loc.flatten() for d in distributions])))
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_perimage(prediction == i, label == i))

    if test_save_path is not None:
        case = case.replace("/", "@")
        np.save(os.path.join(test_save_path, f"{case}_prediction.npy"), prediction.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_label.npy"), label.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_image.npy"), image.astype(np.float32))
    return metric_list


def sample_single_volume_with_distribution(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            label_slice = label[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                label_slice = zoom(label_slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            label_input = torch.from_numpy(label_slice).unsqueeze(0).float().cuda()
            net.train()
            with torch.no_grad():
                outputs, distributions = net(input, None)
                if test_save_path is not None:
                    for level, distribution in enumerate(distributions):
                        np.save(os.path.join(test_save_path, f"level{ind}_" + case + f"_depth{level}_loc.npy"), distribution.loc.cpu().detach().numpy())
                        np.save(os.path.join(test_save_path, f"level{ind}_" + case + f"_depth{level}_scale.npy"), distribution.scale.cpu().detach().numpy())
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.train()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        for level in range(len(prediction)):
            np.save(os.path.join(test_save_path, f"level{level}_" + case + "_prediction.npy"), np.rot90(prediction.astype(np.float32)[level].squeeze(), k=3))
    return metric_list


