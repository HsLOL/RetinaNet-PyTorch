from pprint import pprint
import numpy as np


class Config:
    # lr
    lr = 1e-4

    # warmup setting
    warmup_action = True  # False: not using warmup True: using warmup
    warmup_lr = 1e-5
    warmup_epoch = 2

    # setting
    image_size = 448  # input image resolution
    keep_ratio = False

    classes = ('ship',)  # if only one class, should add ','
    data_path = r''  # absolute data root path
    output_path = r''  # absolute model output path

    inshore_data_path = r''  # absolute Inshore data path
    offshore_data_path = r''  # absolute Offshore data path

    # training setting
    Evaluate_val_start = 1
    Evaluate_train_start = 1
    save_interval = 2  # save weight file
    val_interval = 2  # check the val result on validation set

    # label assignment (Max IoU Assigner)
    pos_iou_thr = 0.5
    neg_iou_thr = 0.4
    min_pos_iou = 0
    low_quality_match = True  # Low quality match

    # anchor setting
    base_size = 4
    ratios = np.array([0.5, 1., 2.])
    # ratios = np.array([1])
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    # scales = np.array([2 ** 0])

    # reg loss func
    # support method = ['giou', 'smooth_l1', 'l1_loss']
    loss_func = 'smooth_l1'

    # nms setting
    nms_thr = 0.5  # for mAP@.5 NMS threshold [nms_thr = 0.5]
    score_thr = 0.05  # follow mmdet RetinaNet settings [score_thr = 0.05]

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def _print_cfg(self):
        pprint(self._state_dict())


cfg = Config()

if __name__ == '__main__':
    cfg._print_cfg()

    # Check
    print(f'the number of the classes: {len(cfg.classes)}')
    for cat in cfg.classes:
        print(