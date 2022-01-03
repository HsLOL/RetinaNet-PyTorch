import os
import torch.utils.data as data
from config import cfg
from pycocotools.coco import COCO
from utils.augment import *


class SSDDataset(data.Dataset):
    def __init__(self, root_dir=cfg.data_path, set_name='train', transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        if transform is True:
            print('[Info]: ===== Using Data Augmentation =====')
        if transform is False:
            print('[Info]: ===== Not using Data Augmentation =====')

        self.coco = COCO(os.path.join(root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.classes = cfg.classes
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image, image_name = self._load_image(index)
        annot = self._load_annotation(index)
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        if self.transform:
            """the transform not include resize function,
               resize will be realized in the collater.py. """
            transform = Augment([
                # HSV(0.5, 0.5, p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Blur(1.3, p=0.5),
                Noise(0.02, p=0.2)
            ], box_mode='xyxy')
            sample = transform(sample)
        return sample

    def _load_annotation(self, index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[index], iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['obj_width'] < 1 or a['obj_height'] < 1:
                continue

            annotation = np.zeros((1, 5))

            # a['bbox'] : x1, y1, x2, y2
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return image, image_info['file_name']
