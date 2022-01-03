import random
import numpy as np
import cv2
import imgaug.augmenters as iaa


class Augment(object):
    def __init__(self, augmentations, probs=1, box_mode=None):
        self.augmentations = augmentations
        self.probs = probs
        self.mode = box_mode

    def __call__(self, sample):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                sample = augmentation(sample, self.mode)
        return sample


class HSV(object):
    """
    Arguments:
        saturation: float 0.5
        brightness: float 0.5
        p: probability float 0.5
    """
    def __init__(self, saturation=0, brightness=0, p=0.):
        self.saturation = saturation
        self.brightness = brightness
        self.p = p

    def __call__(self, sample, mode=None):
        image = sample['image']
        annot = sample['annot']
        image_name = sample['image_name']
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


class Blur(object):
    """
    Arguments:
        sigma: float 1.3
        p: probability float 0.5
    """
    def __init__(self, sigma=0, p=0.):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample, mode=None):
        image, annot, image_name = sample['image'], sample['annot'], sample['image_name']
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0, self.sigma))
            image = blur_aug.augment_image(image)
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


class Contrast(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, sample, mode=None):
        image, annot, image_name = sample['image'], sample['annot'], sample['image_name']
        if random.random() < self.p:
            contrast_aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            image = contrast_aug.augment_image(image)
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


class Noise(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, sample, mode=None):
        image, annot, image_name = sample['image'], sample['annot'], sample['image_name']
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            image = noise_aug.augment_image(image)
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


class HorizontalFlip(object):
    """
    Arguments:
        p: probability float 0.5
    """
    def __init__(self, p=0.):
        self.p = p

    def __call__(self, sample, mode=None):
        image, annot, image_name = sample['image'], sample['annot'], sample['image_name']
        if random.random() < self.p:
            image = np.fliplr(image)
            rows, cols, channel = image.shape
            if mode == 'xyxy':  # [x1, y1, x2, y2]
                x1 = annot[:, 0].copy()
                x2 = annot[:, 2].copy()

                x_temp = x1.copy()
                annot[:, 0] = cols - x2
                annot[:, 2] = cols - x_temp
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


class VerticalFlip(object):
    """
    Arguments:
        p: probability float 0.5
    """

    def __init__(self, p=0.):
        self.p = p

    def __call__(self, sample, mode=None):
        image, annot, image_name = sample['image'], sample['annot'], sample['image_name']
        if random.random() < self.p:
            image = np.flipud(image)
            rows, cols, channel = image.shape
            if mode == 'xyxy':  # [x1, y1, x2, y2]
                y1 = annot[:, 1].copy()
                y2 = annot[:, 3].copy()

                y_temp = y1.copy()
                annot[:, 1] = rows - y2
                annot[:, 3] = rows - y_temp
        sample = {'image': image, 'annot': annot, 'image_name': image_name}
        return sample


if __name__ == '__main__':
    image = cv2.imread('../raw.jpeg')
    labels = np.array([243, 40, 984, 550])  # [x1, y1, x2, y2]
    color = [255, 255, 0]
    # cv2.rectangle(image, (243, 40), (984, 550), color=[255, 255, 0], thickness=2)
    # cv2.imshow('oribox', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 1. HSV
    # hsv = HSV(0.5, 0.5, 0.5)
    # hsv_image, hsv_label = hsv(image, labels)
    # point1 = (hsv_label[0], hsv_label[1])
    # point2 = (hsv_label[2], hsv_label[3])
    # cv2.rectangle(hsv_image, point1, point2, color=color, thickness=2)
    # cv2.imshow('drawbox', hsv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 2. Blur
    # transform = Augment([Blur(10, 0.5)], box_mode='xyxy')
    # trans_img, trans_label = transform(image, labels)
    # point1 = (trans_label[0], trans_label[1])
    # point2 = (trans_label[2], trans_label[3])
    # cv2.rectangle(trans_img, point1, point2, color=color, thickness=2)
    # cv2.imshow('drawbox', trans_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3. Contrast  效果比较随机,用当前值多试几次,效果差别很大
    # transform = Augment([Contrast(1, 0.5)], box_mode='xyxy')
    # trans_img, trans_label = transform(image, labels)
    # point1 = (trans_label[0], trans_label[1])
    # point2 = (trans_label[2], trans_label[3])
    # cv2.rectangle(trans_img, point1, point2, color=color, thickness=2)
    # cv2.imshow('drawbox', trans_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 4. Noise  效果比较随机,用当前值多试几次,效果差别很大
    # transform = Augment([Noise(0.5, 0.5)], box_mode='xyxy')
    # trans_img, trans_label = transform(image, labels)
    # point1 = (trans_label[0], trans_label[1])
    # point2 = (trans_label[2], trans_label[3])
    # cv2.rectangle(trans_img, point1, point2, color=color, thickness=2)
    # cv2.imshow('drawbox', trans_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()