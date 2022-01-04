import os
import shutil
from detect import im_detect
from config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import cv2
import json
from utils.map import eval_mAP
from InShore_OffShore_EvalUtils import In_Off_evaluate


def check_status(args):
    if args.evaluate is True:
        assert (args.FPS is False) and (args.Inshore is False) and (args.Offshore is False), \
            'If args.evaluate is True, other args\' parameters must be False.'

    if args.FPS is True:
        assert (args.evaluate is False) and (args.Inshore is False) and (args.Offshore is False), \
            'If args.FPS is True, other args\' parameters must be False.'

    if args.Inshore or args.Offshore:
        assert (args.evaluate is False) and (args.FPS is False), \
            'If args.Inshore or args.Offshore is True, other args\' parameters must be False.'


def coco_eval_map(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    # print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    """ coco_eval.stats:
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
    """
    return coco_eval.stats[1]


def voc_evaluate(model=None,
                 target_size=None,
                 test_path=None,
                 conf=None,
                 dataset=None,
                 device=None,
                 mode=None):
    evaluate_dir = mode + '_evaluate'
    out_dir = os.path.join(cfg.output_path, evaluate_dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Step1. Collect detect result for per image or get predict result
    for image_name in tqdm(os.listdir(os.path.join(cfg.data_path, mode))):
        image_path = os.path.join(cfg.data_path, mode, image_name)
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model=model,
                         image=image,
                         target_sizes=target_size,
                         use_gpu=True,
                         conf=conf,
                         device=device)

        # Step2. Write per image detect result into per txt file
        # line = cls_name score x1 y1 x2 y2
        img_ext = image_name.split('.')[-1]
        with open(os.path.join(out_dir, image_name.replace(img_ext, 'txt')), 'w') as f:
            for det in dets:
                cls_ind = int(det[0])
                cls_socre = det[1]
                pred_box = det[2:6]
                line = str(cfg.classes[cls_ind]) + ' ' + str(cls_socre) + ' ' + str(pred_box[0]) + ' ' + str(pred_box[1]) +\
                    ' ' + str(pred_box[2]) + ' ' + str(pred_box[3]) + '\n'
                f.write(line)

    # Step3. Calculate Precision, Recall, mAP, plot PR Curve
    mAP, result_dict = eval_mAP(gt_root_dir=cfg.data_path,
                                test_path=test_path,
                                eval_root_dir=os.path.join(cfg.output_path, evaluate_dir),
                                use_07_metric=False,
                                thres=0.5,
                                conf=conf)

    # Print detection results
    print(f'------------- VOC Evaluation --------------')
    print(f'Current mAP:{mAP}')
    for cat in result_dict.keys():
        cat_dictory = result_dict[cat]
        print(f"{cat}:\t precision={cat_dictory['precision']}\trecall={cat_dictory['recall']}\tf1={cat_dictory['f1']}\t"
              f"AP={cat_dictory['AP']}")
    return [mAP, result_dict]


def coco_evaluate(model=None,
                  target_size=None,
                  test_path=None,
                  conf=None,
                  dataset=None,
                  mode=None,
                  device=None):
    evaluate_dir = 'evaluate'  # relative path
    out_dir = os.path.join(cfg.output_path, evaluate_dir, f'{dataset}_{mode}_detection_results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # todo: here is detect result from model
    VAL_GT = test_path
    VAL_IMAGES = os.path.join(cfg.data_path, mode)
    MAX_IMAGES = 10000
    print('\n')
    COCO_GT = COCO(VAL_GT)
    IMAGE_IDS = COCO_GT.getImgIds()[:MAX_IMAGES]
    results = []
    for img_id in tqdm(IMAGE_IDS):
        image_info = COCO_GT.loadImgs(img_id)[0]
        image_path = os.path.join(VAL_IMAGES, image_info['file_name'])
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model=model,
                         image=image,
                         target_sizes=target_size,
                         use_gpu=True,
                         conf=conf,
                         device=device)

        # Step1. Collect detect result for per image or get predict result
        for det in dets:
            cls_ind = int(det[0])
            cls_score = det[1]
            pred_box = det[2:6]

            image_result = {
                'image_id': img_id,
                'category_id': cls_ind + 1,
                'score': cls_score,
                'bbox': pred_box.tolist()}

            results.append(image_result)

    # Step2. Write the pred result into json file
    if len(results) == 0:
        print('[Info]: Current model dont\'t detect anything, Don\'t create json file or eval this pth file.')
        return 0, 0, 0, 0
    else:
        json_file = f'{dataset}_{mode}_result.json'
        json_file_path = os.path.join(out_dir, json_file)
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        else:
            json.dump(results, open(json_file_path, 'w'), indent=4)

        # Step3. Calculate map
        print('\n------------ COCO EVAL -----------\n')
        mAP = coco_eval_map(COCO_GT, IMAGE_IDS, json_file_path)
        return 0, 0, mAP, 0


def evaluate(target_size=None,
             test_path=None,  # relative path of test GT
             eval_method=None,
             model=None,
             conf=None,
             device=None,
             mode=None):
    if test_path.endswith('.json'):
        test_path = os.path.join(cfg.data_path,  test_path)

    model.eval()

    if eval_method == 'coco':
        """COCO format evaluate method.
        Note:
        Now COCO format evaluate method not support F1 socre etc method
           results = 0, 0, mAP, 0
        """
        print(f'\n[Info]: Using coco_evaluate() function.')
        results = coco_evaluate(model=model,
                                target_size=target_size,
                                test_path=test_path,
                                conf=conf,
                                dataset=eval_method,
                                mode=mode,
                                device=device)
    elif eval_method == 'voc':
        """VOC format evaluate method.
           results = [mAP, result_dict]."""
        print(f'\n[Info]: Using voc_evaluate() function.')
        results = voc_evaluate(model=model,
                               target_size=target_size,
                               test_path=test_path,
                               conf=conf,
                               dataset=eval_method,
                               device=device,
                               mode=mode)
    else:
        raise NotImplementedError(f'eval method {eval_method} Unsupported !')
    return results


if __name__ == '__main__':
    import argparse
    import torch
    from config import cfg
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--Dataset', type=str, default='SSDD')
    parser.add_argument('--single_image', type=str, default='train/000025.jpg',
                        help='the relative path of image for test')
    parser.add_argument('--target_size', type=int, default=448)
    parser.add_argument('--chkpt', type=str, default='54_1595.pth', help='the checkpoint file of the trained model.')

    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--FPS', type=bool, default=False, help='Check the FPS of the Model.')

    parser.add_argument('--Offshore', type=bool, default=False, help='Evaluate the Offshore targets performance')
    parser.add_argument('--Inshore', type=bool, default=False, help='Evaluate the Inshore targets performance.')

    args = parser.parse_args()

    check_status(args)

    from models.model import RetinaNet
    model = RetinaNet(backbone='resnet50', loss_func=cfg.loss_func, pretrained=False)

    checkpoint = os.path.join(cfg.output_path, 'checkpoints', args.chkpt)

    # from checkpoint load model weight file
    # model weight
    chkpt = torch.load(checkpoint, map_location='cpu')
    pth = chkpt['model']
    model.load_state_dict(pth)
    model.cuda(device=args.device)

    """The following codes is used to Debug eval() function."""
    if args.evaluate:
        results = evaluate(target_size=[args.target_size],
                           test_path='ground-truth',
                           eval_method='voc',
                           model=model,
                           conf=0.25,
                           device=args.device,
                           mode='val')
        print(results)

    """The following codes are used to calculate FPS of model."""
    if args.FPS:
        times = 50  # 50 is enough to balance some additional times for IO
        image_path = os.path.join(cfg.data_path, args.single_image)
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        model.eval()
        t1 = time.time()
        for _ in range(times):
            dets = im_detect(model=model,
                             image=image,
                             target_sizes=[args.target_size],
                             use_gpu=True,
                             conf=0.25,
                             device=args.device)
        t2 = time.time()
        tact_time = (t2 - t1) / times
        print(f'{tact_time} seconds, {1 / tact_time} FPS, Batch_size = 1')

    if args.Offshore or args.Inshore:
        In_Off_dict = {'Offshore': {'flag': False,
                                    'test_path': 'ground-truth'},
                       'Inshore': {'flag': False,
                                   'test_path': 'ground-truth'}}

        if args.Offshore:
            print('[Info]: Ready to evaluate model with Offshore Targets.')
            In_Off_dict['Offshore']['flag'] = True
        if args.Inshore:
            print('[Info]: Ready to evaluate model with Inshore Targets.')
            In_Off_dict['Inshore']['flag'] = True

        results_dict = In_Off_evaluate(target_size=[args.target_size],
                                       dataset=args.Dataset,
                                       model=model,
                                       conf=0.25,
                                       device=args.device,
                                       **In_Off_dict)
        print(f'\n================ Summary Inshore and Offshore Result ============\n')
        for key, value in results_dict.items():
            if key == 'inshore':
                print(f'[Inshore Evaluate Results]')
                mAP = value[0]
                print(f'Current mAP:{mAP}')
                result_dict = value[1]
                for cat in result_dict.keys():
                    cat_dictory = result_dict[cat]
                    print(f"{cat}:\t precision={cat_dictory['precision']}\trecall={cat_dictory['recall']}"
                          f"\tf1={cat_dictory['f1']}\t"
                          f"AP={cat_dictory['AP']}")
            if key == 'offshore':
                print(f'\n[Offshore Evaluate Results]')
                mAP = value[0]
                print(f'Current mAP:{mAP}')
                result_dict = value[1]
                for cat in result_dict.keys():
                    cat_dictory = result_dict[cat]
                    print(f"{cat}:\t precision={cat_dictory['precision']}\trecall={cat_dictory['recall']}"
                          f"\tf1={cat_dictory['f1']}\t"
                          f"AP={cat_dictory['AP']}")
