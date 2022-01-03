import random
import numpy as np
import torch
import os
import time
import torch.nn as nn


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        deterministic is set True if use torch.backends.cudnn.deterministic
        Default is False.
    """
    print(f'[Info]: Set random seed to {seed}, deterministic: {deterministic}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_in_gt_and_in_center_info(priors, gt_bboxes, num_level_anchors, center_sampling=False, center_radius=2.5):
    """
    Args:
        center_radius: used in center sampling.
        priors: anchor box in anchor-based method [xmin, ymin, xmax, ymax].
        gt_bboxes: gt box [xmin, ymin, xmax, ymax].
    """

    num_gt = gt_bboxes.size(0)
    center_x = (priors[:, 0] + priors[:, 2]) / 2.0
    center_y = (priors[:, 1] + priors[:, 3]) / 2.0
    repeated_x = center_x.unsqueeze(1).repeat(1, num_gt)
    repeated_y = center_y.unsqueeze(1).repeat(1, num_gt)

    # Condition1: if anchor centers in gt boxes area
    l_ = repeated_x - gt_bboxes[:, 0]
    t_ = repeated_y - gt_bboxes[:, 1]
    r_ = gt_bboxes[:, 2] - repeated_x
    b_ = gt_bboxes[:, 3] - repeated_y

    deltas = torch.stack([l_, t_, r_, b_], dim=1)
    is_in_gts = deltas.min(dim=1).values > 0  # (num_anchor, num_gt)
    is_in_gts_all = is_in_gts.sum(dim=1) > 0

    # Condition2: if anchor centers in gt boxes center
    if center_sampling is True:
        device = priors.device
        strides = [8, 16, 32, 64, 128]
        stride_x = torch.ones(num_level_anchors[0]).cuda(device=device) * strides[0]
        stride_y = torch.ones(num_level_anchors[0]).cuda(device=device) * strides[0]

        for idx in range(1, len(strides)):
            tensor = torch.ones(int(num_level_anchors[idx])).cuda(device=device) * strides[idx]
            stride_x = torch.cat((stride_x, tensor))
            stride_y = torch.cat((stride_y, tensor))

        repeated_stride_x = stride_x.unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = stride_y.unsqueeze(1).repeat(1, num_gt)

        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0

        ct_box_l = gt_cxs - center_radius * repeated_stride_x
        ct_box_t = gt_cys - center_radius * repeated_stride_y
        ct_box_r = gt_cxs + center_radius * repeated_stride_x
        ct_box_b = gt_cys + center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0  # (num_anchor, num_gt)
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # anchor whose centers in boxes or centers
        is_in_boxes_or_centers = (is_in_gts & is_in_cts)

        # is_in_gts_or_center.shape(num_anchor), is_in_boxes_or_centers.shape(num_anchor, num_gt)
        return is_in_boxes_or_centers, is_in_gts_or_centers

    return is_in_gts, is_in_gts_all  # is_in_gts.shape(num_anchor, num_gt), is_in_gts_all.shape(num_anchor)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def save_checkpoint(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))
    print(f'checkpoint: {name} has saved.')


def write_log_txt(strings, output_path, step, mode=None):
    assert not(mode == None), '[Error]: Must specify the train/val mode to write log txt!'
    t = time.localtime()
    log_name = f'{str(t.tm_mon)}_{str(t.tm_mday)}.txt'
    with open(os.path.join(output_path, log_name), 'a') as f:
        line = f'[{mode} / {step}]:  Total_loss: {str(float(strings[0]))} Cls_loss: {str(float(strings[1]))} Reg_loss: {str(float(strings[2]))}\n'
        f.write(line)


def show_cfg(cfg):
    print('================ Show Cfg ===============')
    cfg._print_cfg()


def show_args(args):
    print('=============== Show Args ===============')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model

    num_params = sum(x.numel() for x in model.parameters())
    num_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)

    if report == 'full':
        print('%5s $40s %9s %12s %20s %10s %10s' %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))

        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    if report == 'summary':
        print('\nModel Summary: %g layers, %g parameters, %g gradients' %
              (len(list(model.parameters())), num_params, num_gradients))

    if report == 'params':
        # check the number of the learnable params
        count = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad is True:
                # print(name)
                count += 1
        print(f'[Info]: {count} layers have learnable params')


def pretty_print(num_params, units=None, precision=2):
    if units is None:
        if num_params // 10**6 > 0:
            print(f'[Info]: Model Params = {str(round(num_params / 10**6, precision))}' + ' M')
        elif num_params // 10**3:
            print(f'[Info]: Model Params = {str(round(num_params / 10**3, precision))}' + ' k')
        else:
            print(f'[Info]: Model Params = {str(num_params)}')


def count_param(model, units=None, precision=2):
    """Count Params"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pretty_print(num_params)




def calc_hbb_iou(a, b, eps=1e-6, mode='iou', is_aligned=None):
    # a(anchor) [boxes, (x1, y1, x2, y2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]
    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}.'

    rows = a.size(-2)
    cols = b.size(-2)
    if is_aligned:
        assert rows == cols, f'rows is not equal to cols in calc_hbb_iou func.'

    area1 = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area2 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    if is_aligned:
        lt = torch.max(a[..., :2], b[..., :2])
        rb = torch.min(a[..., 2:], b[..., 2:])

        wh = torch.clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            raise NotImplementedError(f'Unsupported {mode} in calc_hbb_iou func !')

        if mode is 'giou':
            enclosed_lt = torch.min(a[..., :2], b[..., :2])
            enclosed_rb = torch.max(a[..., 2:], b[..., 2:])
    else:
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = torch.clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            raise NotImplementedError(f'Unsupported {mode} in calc_hbb_iou func !')

        if mode == 'giou':
            enclosed_lt = torch.min(a[:, None, :2], b[None, :, :2])
            enclosed_rb = torch.max(a[:, None, 2:], b[None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious

    # calculate gious
    enclose_wh = torch.clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def clip_boxes(boxes, ims):
    _, _, h, w = ims.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=w)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=h)
    return boxes


class BoxCoder(object):
    def __init__(self, weights=(1, 1, 1, 1)):
        self.weights = weights

    def encode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=1
        )
        return targets

    def decode(self, boxes, deltas, mode='xywh', wh_ratio_clip=16/1000):
        """only support horizontal bbox decode."""
        # todo: ready to support ctr_clamp, now is only support wh_clamp
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        dx_width = dx * widths
        dy_height = dy * heights

        max_ratio = np.abs(np.log(wh_ratio_clip))

        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        pred_ctr_x = ctr_x + dx_width
        pred_ctr_y = ctr_y + dy_height
        pred_w = dw.exp() * widths
        pred_h = dh.exp() * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            (pred_boxes_x1,
             pred_boxes_y1,
             pred_boxes_x2,
             pred_boxes_y2)).t()

        return pred_boxes
