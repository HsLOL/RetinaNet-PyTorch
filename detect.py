import numpy as np
import torch

from torchvision.transforms import Compose
from datasets.util import Reshape, Rescale, Normailize
from utils.HBB_NMS_GPU.nms.gpu_nms import gpu_nms
from config import cfg


def im_detect(model, image, target_sizes, use_gpu=True, conf=None, device=None):
    if isinstance(target_sizes, int):
        target_sizes = [target_sizes]
    if len(target_sizes) == 1:
        return single_scale_detect(model, image, target_sizes[0], use_gpu=use_gpu, conf=conf, device=device)
    else:
        ms_dets = None
        for ind, scale in enumerate(target_sizes):
            cls_dets = single_scale_detect(model, image, target_size=scale, use_gpu=use_gpu, conf=conf, device=device)
            if cls_dets.shape[0] == 0:
                continue
            if ms_dets is None:
                ms_dets = cls_dets
            else:
                ms_dets = np.vstack((ms_dets, cls_dets))
        if ms_dets is None:
            return np.zeros((0, 7))
        cls_dets = np.hstack((ms_dets[:, 2:7], ms_dets[:, 1][:, np.newaxis])).astype(np.float32, copy=False)
        keep = gpu_nms(cls_dets, 0.1)
        return ms_dets[keep, :]


def single_scale_detect(model, image, target_size, use_gpu=True, conf=None, device=None):
    im, im_scales = Rescale(target_size=target_size, keep_ratio=cfg.keep_ratio)(image)
    im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)

    # Modify Here
    if use_gpu and torch.cuda.is_available():
        if next(model.parameters()).is_cuda:
            im = im.cuda(device=device)
        else:
            model, im = model.cuda(device=device), im.cuda(device=device)
    with torch.no_grad():
        scores, classes, boxes = model(im, test_conf=conf)
    scores = scores.data.cpu().numpy()
    classes = classes.data.cpu().numpy()
    boxes = boxes.data.cpu().numpy()  # boxes[pre_box[x1, y1, x2, y2], anchor[x1, y1, x2, y2]]
    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:
        boxes[:, 4:8] = boxes[:, 4:8] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)  # cls_dets = [cls, score, pred[x1, y1, x2, y2], anchor[x1, y1, x2, y2]]
    keep = np.where(classes < len(cfg.classes))[0]  # exclude bg class
    return cls_dets[keep, :]
