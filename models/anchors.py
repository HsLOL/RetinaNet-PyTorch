import numpy as np
import torch
import torch.nn as nn
from config import cfg


class Anchors(nn.Module):
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 base_size=None,
                 ratios=None,
                 scales=None,
                 rotations=None):  # Not implement
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = strides
        self.base_size = base_size
        self.ratios = ratios
        self.scales = scales
        self.rotations = rotations

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        if base_size is None:
            self.base_size = cfg.base_size

        if ratios is None:
            self.ratios = cfg.ratios

        if scales is None:
            self.scales = cfg.scales

        self.num_anchors = len(self.scales) * len(self.ratios)

        print(f'[Info]: anchor ratios: {self.ratios}\tanchor scales: {self.scales}\tbase_size: {self.base_size}')
        print(f'[Info]: number of anchors: {self.num_anchors}')

    @staticmethod
    def generate_anchors(base_size, ratios, scales):
        """generate horizontal anchors"""
        num_anchors = len(ratios) * len(scales)
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios))).T
        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) to (x1, y1, x2, y2)
        anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    @staticmethod
    def shift(img_shape, stride, anchors):
        shift_x = (np.arange(0, img_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, img_shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()  # (shift_x, shift_y, shift_x, shift_y)

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K * A, 4) shifted anchors

        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    def forward(self, images):
        image_shape = np.array(images.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        num_level_anchors = []
        for idx, p in enumerate(self.pyramid_levels):
            base_anchors = self.generate_anchors(
                base_size=self.base_size * self.strides[idx],
                ratios=self.ratios,
                scales=self.scales
            )
            shifted_anchors = self.shift(image_shapes[idx], self.strides[idx], base_anchors)
            num_level_anchors.append(shifted_anchors.shape[0])
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (images.size(0), 1, 1))
        all_anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.is_tensor(images) and images.is_cuda:
            device = images.device
            all_anchors = all_anchors.cuda(device=device)
        return all_anchors, torch.from_numpy(np.array(num_level_anchors)).cuda(device=device)


if __name__ == '__main__':
    anchors = Anchors()
    featuremap_sizes = [(80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
    for level_idx in range(5):
        print(f'# ============================base_anchor{level_idx}========================================= #')
        base_anchor = anchors.generate_anchors(
            base_size=anchors.base_size * anchors.strides[level_idx],
            ratios=anchors.ratios,
            scales=anchors.scales
        )
        print(base_anchor)
        print(f'# ============================shift_anchor{level_idx}========================================= #')
        shift_anchor = anchors.shift(featuremap_sizes[level_idx], anchors.strides[level_idx], base_anchor)
        print(shift_anchor)
