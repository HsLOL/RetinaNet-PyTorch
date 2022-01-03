import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import calc_hbb_iou, BoxCoder
from config import cfg


class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func='smooth_l1'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        self.func = func
        print(f'[Info]: ===== Using {func} Loss & Max IoU assign label assignment =====')

        if func == 'smooth_l1':
            self.criteron = smooth_l1_loss

        elif func == 'giou':
            self.criteron = giou_loss

        elif func == 'mse':
            self.criteron = F.mse_loss()

        elif func == 'l1_loss':
            self.criteron = l1_loss

    def forward(self, classifications, regressions, anchors, annotations, image_names):
        """
        Arguments:
            classifications: model pred
            regressions: model pred
            anchors: mlvl anchors
            refined_anchors: predicted bbox
            annotations: GT [[x1, y1, x2, y2, cls], ...]
        """
        pos_iou_thr = cfg.pos_iou_thr
        neg_iou_thr = cfg.neg_iou_thr
        min_pos_iou = cfg.min_pos_iou
        cls_losses = []
        reg_losses = []
        anchor = anchors[0, :, :]
        batch_size = classifications.shape[0]
        device = classifications[0].device
        for idx in range(batch_size):
            image_name = image_names[idx]
            classification = classifications[idx, :, :]
            regression = regressions[idx, :, :]
            bbox_annotation = annotations[idx]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            num_gts = len(bbox_annotation)  # the number of GT  per image
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:

                cls_losses.append(torch.tensor(0).float().cuda(device=device))
                reg_losses.append(torch.tensor(0).float().cuda(device=device))
                continue

            overlaps = calc_hbb_iou(anchor[:, :], bbox_annotation[:, :4], mode='iou')

            # for each anchor, which gt best overlaps with it
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # for each gt, which anchor best overlaps with it
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

            # -------------- compute the classification loss ------------------------------ #
            cls_targets = (torch.ones_like(classification) * -1).cuda(device=device)
            cls_targets[torch.lt(max_overlaps, neg_iou_thr), :] = 0
            positive_indices = torch.ge(max_overlaps, pos_iou_thr)

            # find out positive index
            # positive_index = torch.nonzero(positive_indices)
            # positive_anchors = anchor[positive_index, :]
            # import cv2, os
            # ori_img = cv2.imread(os.path.join(cfg.data_path, 'train', image_name))
            # for positive_anchor in positive_anchors:
            #     positive_anchor = positive_anchor[0]
            #     point1 = (int(positive_anchor[0]), int(positive_anchor[1]))
            #     point2 = (int(positive_anchor[2]), int(positive_anchor[3]))
            #     cv2.rectangle(ori_img, point1, point2, color=[0, 255, 255], thickness=2)
            # cv2.imwrite(f"_{image_name.replace('tif', 'png')}", ori_img)

            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[argmax_overlaps, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # Low-quality matching
            if cfg.low_quality_match:
                for i in range(num_gts):
                    if gt_max_overlaps[i] >= min_pos_iou:
                        max_iou_inds = overlaps[:, i] == gt_max_overlaps[i]
                        cls_targets[max_iou_inds, assigned_annotations[max_iou_inds, 4].long()] = 1
                        positive_indices = positive_indices | max_iou_inds
                num_positive_anchors = positive_indices.sum()

            alpha_factor = torch.ones_like(cls_targets).cuda(device=device) * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            # bin_cross_entrophy = -(cls_targets * torch.log(classification) +
            #                        (1.0 - cls_targets) * torch.log(1.0 - classification))
            #
            bin_cross_entrophy = -(cls_targets * torch.log(classification + 1e-6) +
                                   (1.0 - cls_targets) * torch.log(1.0 - classification + 1e-6))
            cls_loss = focal_weight * bin_cross_entrophy
            zeros = torch.zeros_like(cls_loss).cuda(device=device)
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, zeros)
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # ---------------------------- compute regression loss -------------------------------- #
            if positive_indices.sum() > 0:
                positive_anchors = anchor[positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                if self.func == 'smooth_l1' or self.func == 'mse' or self.func == 'l1_loss':
                    reg_targets = self.box_coder.encode(positive_anchors, gt_boxes)
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                elif self.func == 'giou':
                    pred_bboxes = self.box_coder.decode(
                        positive_anchors, regression[positive_indices, :], mode='xyxy')
                    reg_loss = self.criteron(pred_bboxes, gt_boxes[:, :4], is_aligned=True)
                reg_losses.append(reg_loss)
            else:
                reg_losses.append(torch.tensor(0).float().cuda(device=device))
        # calculate mean cls loss & mean reg loss of per batch size
        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True)

        return loss_cls, loss_reg


def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight=None):
    diff = torch.abs(inputs - targets)
    if weight is None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    if size_average:
        return loss.mean()
    return loss.sum()


def l1_loss(inputs, targets, size_averate=True):
    loss = torch.abs(inputs - targets)
    if size_averate:
        return loss.mean()
    return loss.sum()


def giou_loss(inputs,
              targets,
              size_average=True,
              weight=None,
              is_aligned=True):
    if weight is None:
        giou = calc_hbb_iou(inputs, targets, mode='giou', is_aligned=is_aligned)
        loss = 1 - giou

    if size_average:
        return loss.mean()
    return loss
