from detect import im_detect
import argparse
from models.model import RetinaNet
import os
from config import cfg
import cv2
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--target_sizes', type=list, default=[448], help='support multi scale detect')
    parser.add_argument('--chkpt', type=str, default='best/best.pth', help='the chkpt file name')
    parser.add_argument('--result_path', type=str, default='show_result', help='the relative path for saving'
                                                                               'ori pic and predicted pic')
    parser.add_argument('--score_thresh', type=float, default=0.6, help='score threshold')
    parser.add_argument('--pic_name', type=str, default='demo1.jpg', help='relative path')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args


def plot_box(image, coord, label=None, score=None, color=None, line_thickness=None, show_name=True, put_label=True):
    t1 = line_thickness or int(round(0.001 * max(image.shape[0:2])))
    color = [0, 255, 255]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(image, c1, c2, color, thickness=t1)

    if put_label:
        if show_name:
            label = label + str('%.2f' % score)
        else:
            label = str('%.2f' % score)
        fontScale = 0.3
        font = cv2.FONT_HERSHEY_COMPLEX
        thickness = 1
        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
        coor1 = c1
        coor2 = c1[0] + t_size[0], c1[1] - t_size[1] - 2
        cv2.rectangle(image, coor1, coor2, [0, 255, 0], -1)  # filled
        cv2.putText(image, label, (coor1[0], coor1[1] - 2), font, fontScale, [0, 0, 0],
                    thickness=thickness, lineType=cv2.LINE_AA)


def show_pred_box(args):
    # create folder
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = RetinaNet(backbone=args.backbone, loss_func='smooth_l1', pretrained=False)
    chkpt_path = os.path.join(cfg.output_path, 'checkpoints', args.chkpt)
    chkpt = torch.load(chkpt_path, map_location='cpu')
    print(f"The current model training {chkpt['epoch']} epoch(s)")
    print(f"The current model mAP: {chkpt['best_fitness']} based on test_conf={cfg.score_thr} & nms_thr={cfg.nms_thr}")

    model.load_state_dict(chkpt['model'])
    model.cuda(device=args.device)
    model.eval()

    image = cv2.cvtColor(cv2.imread(os.path.join(args.result_path, args.pic_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # dets = [cls_index, score, predict_box[x1, y1, x2, y2], anchor[x1, y1, x2, y2]]
    dets = im_detect(model,
                     image,
                     target_sizes=args.target_sizes,
                     use_gpu=True,
                     conf=args.score_thresh,
                     device=args.device)

    # plot predict box
    # red box is the original anchor
    for det in dets:
        cls_index = int(det[0])
        score = float(det[1])
        pred_box = list(map(int, det[2:6]))
        anchor = list(map(int, det[6:10]))

        # plot predict box
        plot_box(image, coord=pred_box, label=cfg.classes[cls_index], score=score, show_name=False, put_label=True)
        # plot which anchor to create predict box
        # cv2.rectangle(image, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color=[255, 255, 0], thickness=2)

    cv2.imwrite(os.path.join(args.result_path, 'predict.png'), image)


if __name__ == '__main__':
    args = get_args()
    if args.score_thresh != cfg.score_thr:
        print('[Info]: score_thresh is not equal to cfg.score_thr')
    show_pred_box(args)
