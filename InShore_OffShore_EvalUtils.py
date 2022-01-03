"""This script is a tool which is used to get InShore and OffShore result(mAP, F1) of SSDD Dataset."""
from config import cfg
import os
import shutil
from tqdm import tqdm
import cv2
from detect import im_detect
from utils.map import eval_mAP


Support_Dataset = ['SSDD']


def wrapper(model=None,
            target_size=None,
            conf=None,
            device=None,
            out_dir=None,
            img_path=None,
            in_off_shore_root_path=None,
            test_path=None,
            evaluate_dir=None):
    # Step1. Collect detect result for per image or get predict result
    for image_name in tqdm(os.listdir(os.path.join(in_off_shore_root_path, img_path))):
        image_path = os.path.join(in_off_shore_root_path, img_path, image_name)
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
                line = str(cfg.classes[cls_ind]) + ' ' + str(cls_socre) + ' ' + str(pred_box[0]) + ' ' + str(
                    pred_box[1]) + \
                       ' ' + str(pred_box[2]) + ' ' + str(pred_box[3]) + '\n'
                f.write(line)

    # Step3. Calculate Precision, Recall, mAP, plot PR Curve
    mAP, result_dict = eval_mAP(gt_root_dir=in_off_shore_root_path,
                                test_path=test_path,
                                eval_root_dir=os.path.join(cfg.output_path, evaluate_dir),
                                use_07_metric=False,
                                thres=0.5)

    # Print detection results
    print(f'------------- VOC Evaluation --------------')
    print(f'Current mAP:{mAP}')
    for cat in result_dict.keys():
        cat_dictory = result_dict[cat]
        print(f"{cat}:\t precision={cat_dictory['precision']}\trecall={cat_dictory['recall']}\tf1={cat_dictory['f1']}\t"
              f"AP={cat_dictory['AP']}")
    return [mAP, result_dict]


def Inshore_Evaluate(model=None,
                     target_size=None,
                     test_path=None,
                     conf=None,
                     dataset=None,
                     device=None):
    evaluate_dir = 'evaluate/Inshore_evaluate'
    out_dir = os.path.join(cfg.output_path, evaluate_dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    inshore_result = wrapper(model=model,
                             target_size=target_size,
                             conf=conf,
                             device=device,
                             out_dir=out_dir,
                             img_path='images',  # relative path of saving inshore/offshore images
                             in_off_shore_root_path=cfg.inshore_data_path,  # inshore and offshore root path
                             test_path=test_path,
                             evaluate_dir=evaluate_dir)
    return inshore_result


def Offshore_Evaluate(model=None,
                      target_size=None,
                      test_path=None,
                      conf=None,
                      dataset=None,
                      device=None):
    evaluate_dir = 'evaluate/Offshore_evaluate'
    out_dir = os.path.join(cfg.output_path, evaluate_dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    offshore_result = wrapper(model=model,
                              target_size=target_size,
                              conf=conf,
                              device=device,
                              out_dir=out_dir,
                              img_path='images',  # relative path of saving inshore/offshore images
                              in_off_shore_root_path=cfg.offshore_data_path,  # inshore and offshore root path
                              test_path=test_path,
                              evaluate_dir=evaluate_dir)
    return offshore_result


def In_Off_evaluate(target_size=None,
                    dataset=None,
                    model=None,
                    conf=None,
                    device=None,
                    **kwargs):

    model.eval()
    if dataset in Support_Dataset:
        print(f'[Info]: Using voc_evaluate() function.')
        for key, value in kwargs.items():
            """kwargs = {'Inshore': {'flag': True, 'test_path': ''},
                         'Offshore': {'flag': True, 'test_path':''}}"""
            if key == 'Inshore':
                inshore_evaluate_flag = value['flag']
                inshore_evaluate_path = value['test_path']
            if key == 'Offshore':
                offshore_evaluate_flag = value['flag']
                offshore_evaluate_path = value['test_path']

        if inshore_evaluate_flag is True:
            print(f'[Info]: Evaluate Model with Inshore targets.')
            inshore_results = Inshore_Evaluate(model=model,
                                               target_size=target_size,
                                               test_path=inshore_evaluate_path,
                                               conf=conf,
                                               dataset=dataset,
                                               device=device)

        if offshore_evaluate_flag is True:
            print(f'[Info]: Evaluate Model with Offshore targets.')
            offshore_results = Offshore_Evaluate(model=model,
                                                 target_size=target_size,
                                                 test_path=offshore_evaluate_path,
                                                 conf=conf,
                                                 dataset=dataset,
                                                 device=device)

        return {'inshore': inshore_results,
                'offshore': offshore_results}
    else:
        raise NotImplementedError(f'Add {dataset} in Support_Dataset.')
