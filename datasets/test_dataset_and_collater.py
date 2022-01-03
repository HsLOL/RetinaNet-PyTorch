"""This script is used to test the punction of custom class Dataset and collater."""
from datasets.SSDD_dataset import SSDDataset
from datasets.collater import Collater
from config import cfg
import random
import numpy as np


if __name__ == '__main__':
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    training_set = SSDDataset(root_dir=cfg.data_path,
                              set_name='train',
                              transform=False)

    """Check some outputs from custom dataset. 
       User can specify the test_idx manually. """
    # test_idxs = np.arange(1, 20)
    # for idx in test_idxs:
    #     single_sample = training_set[idx]
    #     trans_img = single_sample['image']
    #     trans_img = trans_img.astype(np.uint8)
    #     trans_annot = single_sample['annot']
    #     img_name = single_sample['image_name']
    #     for poly in trans_annot:
    #         point1 = (int(poly[0]), int(poly[1]))
    #         point2 = (int(poly[2]), int(poly[3]))
    #         cv2.rectangle(trans_img, point1, point2, [0, 255, 255], 2)
    #     plt.imshow(trans_img)
    #     plt.show()


    """Check some outputs from custom collater.
       1. User can specify the test_idx manually. 
       2. User can visualize scale image result to cancel annotation line (57-65) in collater.py"""
    # test_idxs = [0, 2, 4, 8, 10, 12, 14, 16, 18]
    # batch = [training_set[idx] for idx in test_idxs]
    # collater = Collater(scales=448, keep_ratio=False, multiple=32)
    # result = collater(batch)
    #
    # for image, name in zip(result['image'], result['image_name']):
    #     print(f'The {name} is scaled to size: {image.size()[-2:]}')
