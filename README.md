## :rocket:RetinaNet Horizontal Detector Based PyTorch
This is a horizontal detector **RetinaNet** implementation on remote sensing dataset(SSDD).  
This re-implemented retinanet has the almost the same mAP(iou=.5) with the MMdetection.  
RetinaNet Detector original paper link is [here](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).  
## :star2:Performance of the implemented RetinaNet Detector  
### Detection Performance on Offshore image.
### Detection Performance on Inshore image.
## :dart:Experiment
The SSDD dataset, well-trained retinanet detector, resnet-50 pretrained model on ImageNet, loss curve, evaluation metrics results are below, you could follow my experiment.  
- SSDD dataset [BaiduYun]() `extraction code=`  
- gt labels for eval [BaiduYun]() `extraction code=`  
- well-trained retinanet detector weight file [BaiduYun]() `extraction code=`  
- pre-trained ImageNet resnet-50 weight file [BaiduYun]() `extraction code=`  
- evaluation metrics(iou=.5)
|Batch Size|Input Size|mAP (Mine)|mAP (MMdet)|Model Parameters|
|:--------:|:--------:|:--------:|:---------:|:--------------:|
|32        |416 x 416 |0.89      |0.8891     |32.2 M|
- loss curve
## :boom:Get Started
