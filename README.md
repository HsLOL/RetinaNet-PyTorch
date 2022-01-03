## :rocket:RetinaNet Horizontal Detector Based PyTorch
This is a horizontal detector **RetinaNet** implementation on remote sensing dataset(SSDD).  
This re-implemented retinanet has the almost the same mAP(iou=.5) with the MMdetection.  
RetinaNet Detector original paper link is [here](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).  
## :star2:Performance of the implemented RetinaNet Detector  
### Detection Performance on Offshore image.  
<img src="https://github.com/HsLOL/RetinaNet-PyTorch/tree/master/pics/inshore_predict.png" width="300" height="300"/>
### Detection Performance on Inshore image.  
<img src="https://github.com/HsLOL/RetinaNet-PyTorch/tree/master/pics/offshore_predict.png" width="300" height="300"/>
## :dart:Experiment
The SSDD dataset, well-trained retinanet detector, resnet-50 pretrained model on ImageNet, loss curve, evaluation metrics results are below, you could follow my experiment.  
- SSDD dataset [BaiduYun]() `extraction code=`  
- gt labels for eval [BaiduYun]() `extraction code=`  
- well-trained retinanet detector weight file [BaiduYun]() `extraction code=`  
- pre-trained ImageNet resnet-50 weight file [BaiduYun]() `extraction code=`  
- evaluation metrics(iou=.5)  

| Batch Size | Input Size | mAP (Mine) | mAP (MMdet) | Model Parameters |  
|:----------:|:----------:|:----------:|:-----------:|:----------------:|  
|32          | 416 x 416  | 0.89       | 0.8891      | 32.2 M           |  
- loss curve
## :boom:Get Started  
### Installation
#### A. Install requirements:
```
conda create -n retinanet python=3.7
conda activate retinanet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt  

Note: If you meet some troubles about installing environment, you can see the check.txt for more details.
```
#### B. Install nms module:
```
cd utils/nms
make
```
## Demo
you should download the trained weight file.
```
# run the simple inference script to get detection result.
python show.py
```
## Train
### A. Prepare dataset
you should structure your dataset files like this.
```
# dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, coco2017
datasets/
    -coco2017/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```
### B. Manual set project's hyper parameters
you should manual set projcet's hyper parameters in `config.py`
```

```
### C. Train RetinaNet detector on a custom dataset with pretrianed resnet-50 from scratch
```
```
## Evaluation
