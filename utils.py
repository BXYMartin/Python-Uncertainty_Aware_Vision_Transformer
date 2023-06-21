import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from medpy.metric import jc


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
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect) / (z_sum + y_sum + smooth)
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

def ncc(a,v, zero_norm=True):

    a = a.flatten()
    v = v.flatten()

    if np.isclose(np.std(a), 0, 1e-8) and np.isclose(np.std(v), 0, 1e-8):
        return 1
    if np.std(v) == 0:
        return np.correlate(a,v)

    if zero_norm:

        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)

    else:

        a = (a) / (np.std(a) * len(a))
        v = (v) / np.std(v)

    return np.correlate(a,v)
    
def generalised_energy_distance(sample_arr, gt_arr, nlabels=1):

    def dist_fct(m1, m2):
        label_range = range(nlabels)
        per_label_iou = []
        for lbl in [nlabels]:

            # assert not lbl == 0  # tmp check
            m1_bin = (m1 == lbl)*1
            m2_bin = (m2 == lbl)*1

            if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
                per_label_iou.append(1)
            elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
                per_label_iou.append(0)
            else:
                per_label_iou.append(jc(m1_bin, m2_bin))

        # print(1-(sum(per_label_iou) / nlabels))

        return 1-(sum(per_label_iou) / nlabels)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            # print(dist_fct(sample_arr[i,...], gt_arr[j,...]))
            d_sy.append(dist_fct(sample_arr[i,...], gt_arr[j,...]))

    for i in range(N):
        for j in range(N):
            # print(dist_fct(sample_arr[i,...], sample_arr[j,...]))
            d_ss.append(dist_fct(sample_arr[i,...], sample_arr[j,...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(dist_fct(gt_arr[i,...], gt_arr[j,...]))

    return (2./(N*M))*sum(d_sy) - (1./N**2)*sum(d_ss) - (1./M**2)*sum(d_yy)


# import matplotlib.pyplot as plt
def variance_ncc_dist(sample_arr, gt_arr):

    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
        if np.isclose(m_samp.max(), 0, 1e-8) or np.isclose(m_gt.max(),0, 1e-8):
            return 0
        log_samples = np.log(m_samp + eps)
        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y
    :return: 
    """

    mean_seg = np.mean(sample_arr, axis=0)

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    sX = sample_arr.shape[1]
    sY = sample_arr.shape[2]

    E_ss_arr = np.zeros((N,sX,sY))
    for i in range(N):
        E_ss_arr[i,...] = pixel_wise_xent(sample_arr[i,...], mean_seg)
        # print('pixel wise xent')
        # plt.imshow( E_ss_arr[i,...])
        # plt.show()

    E_ss = np.mean(E_ss_arr, axis=0)

    E_sy_arr = np.zeros((M,N, sX, sY))
    for j in range(M):
        for i in range(N):
            E_sy_arr[j,i, ...] = pixel_wise_xent(sample_arr[i,...], gt_arr[j,...])

    E_sy = np.mean(E_sy_arr, axis=1)

    ncc_list = []
    for j in range(M):

        ncc_list.append(ncc(E_ss, E_sy[j,...]))

    return (1/M)*sum(ncc_list)

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

def calculate_metric_single_image(pred_all, gt_all):
    # pred_all = pred_all[0:1, :, :]
    ged = float(generalised_energy_distance(pred_all, gt_all))
    ncc = float(variance_ncc_dist(pred_all, gt_all))
    
    return ged, ncc



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


def test_multiple_image(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, repeat=12):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    batch_size = label.shape[0]
    all_prediction = np.zeros((repeat * batch_size, label.shape[1], label.shape[2]))
    for repeat_index in range(repeat):
        if len(image.shape) == 3:
            prediction = np.zeros_like(label)
            # for ind in range(image.shape[0]):
            slice = image[:, :, :]
            label_slice = label[:, :, :]
            x, y = slice.shape[1], slice.shape[2]
            if x != patch_size[0] or y != patch_size[1]:
                slice_new = np.zeros((slice.shape[0], *patch_size))
                label_slice_new = np.zeros((label_slice.shape[0], *patch_size))
                for ind in range(slice.shape[0]):
                    slice_new[ind] = zoom(slice[ind], (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                for ind in range(label_slice.shape[0]):
                    label_slice_new[ind] = zoom(label_slice[ind], (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                slice = slice_new
                label_slice = label_slice_new
            input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
            label_input = torch.from_numpy(label_slice).float().cuda()
            net.train()
            with torch.no_grad():
                outputs, distributions = net(input, label_input)
                #print(torch.mean(torch.concat([d.loc.flatten() for d in distributions])))
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = np.zeros((out.shape[0], x, y))
                    for ind in range(out.shape[0]):
                        pred[ind] = zoom(out[ind], (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction = pred
        else:
            input = torch.from_numpy(image).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.train()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(net(input, None)[0], dim=1), dim=1).squeeze(0)
                prediction = out.cpu().detach().numpy()
        all_prediction[repeat_index*batch_size:(repeat_index+1)*batch_size] = prediction
    metric_list = []
    metric_list.append(calculate_metric_single_image(all_prediction, label))

    if test_save_path is not None:
        case = case.replace("/", "@")
        np.save(os.path.join(test_save_path, f"{case}_prediction.npy"), prediction.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_label.npy"), label.astype(np.float32))
        np.save(os.path.join(test_save_path, f"{case}_image.npy"), image.astype(np.float32))
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


