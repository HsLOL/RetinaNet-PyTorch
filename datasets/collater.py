import torch
import numpy as np
import numpy.random as npr
from datasets.util import Rescale, Normailize, Reshape
from torchvision.transforms import Compose


class Collater(object):
    """ Data Augmentation is realized in the dataset.py and
        Resize, Normalize, Reshape, etc function is realized in the class Collater. """
    def __init__(self, scales, keep_ratio=False, multiple=32):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)

        self.keep_ratio = keep_ratio
        self.multiple = multiple

    def __call__(self, batch):
        # todo: how to realize multi_scale training
        random_scale_inds = npr.randint(0, high=len(self.scales))
        target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)
        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)

        # custom transform function here
        transform = Compose([Normailize(), Reshape(unsqueeze=False)])

        images = [data['image'] for data in batch]
        annots = [data['annot'] for data in batch]
        image_names = [data['image_name'] for data in batch]

        batch_size = len(images)
        max_width, max_height = -1, -1
        for i in range(batch_size):
            im, _ = rescale(images[i])
            height, width = im.shape[0], im.shape[1]
            max_width = width if width > max_width else max_width
            max_height = height if height > max_height else max_height

        padded_ims = torch.zeros(batch_size, 3, max_height, max_width)

        num_params = annots[0].shape[-1]
        max_num_annots = max(annot.shape[0] for annot in annots)
        padded_annots = torch.ones(batch_size, max_num_annots, num_params) * -1
        for i in range(batch_size):
            im, annot = images[i], annots[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)
            if num_params == 5:
                annot[:, :4] = annot[:, :4] * im_scale
            padded_annots[i, :annot.shape[0], :] = torch.from_numpy(annot)

            # # vis img and annot rescale result
            # import cv2
            # import matplotlib.pyplot as plt
            # img = im.astype(np.uint8)
            # for poly in annot:
            #     point1 = (int(poly[0]), int(poly[1]))
            #     point2 = (int(poly[2]), int(poly[3]))
            #     cv2.rectangle(img, point1, point2, color=[0, 255, 255], thickness=2)
            # plt.imshow(img)
            # plt.show()

        return {'image': padded_ims, 'annot': padded_annots, 'image_name': image_names}