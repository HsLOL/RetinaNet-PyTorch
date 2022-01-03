import torch
import torch.nn as nn
from config import cfg
from models.anchors import Anchors
from models.fpn import FPN, LastLevelP6_P7
from models import resnet
from models.heads import CLSBranch, REGBranch, CLSHead, REGHead
from models.losses import IntegratedLoss
from utils.utils import BoxCoder
from utils.utils import clip_boxes
from utils.HBB_NMS_GPU.nms.gpu_nms import gpu_nms


class RetinaNet(nn.Module):
    def __init__(self, backbone='resnet50', loss_func='smooth_l1', pretrained=True):
        super(RetinaNet, self).__init__()
        self.num_class = len(cfg.classes)
        self.num_regress = 4
        self.anchor_generator = Anchors()
        self.num_anchors = self.anchor_generator.num_anchors
        self.pretrained = pretrained
        self.init_backbone(backbone)
        self.cls_branch_num_stacked = 4

        self.fpn = FPN(
            in_channel_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6_P7(in_channels=256,
                                      out_channels=256,
                                      init_method='xavier_init'),  # in_channels: 1) 2048 on C5, 2) 256 on P5
            init_method='xavier_init'
            )

        self.cls_branch = CLSBranch(
            in_channels=256,
            feat_channels=256,
            num_stacked=self.cls_branch_num_stacked,
            init_method='normal_init'
        )

        self.cls_head = CLSHead(
            feat_channels=256,
            num_anchors=self.num_anchors,
            num_classes=self.num_class
        )

        self.reg_branch = REGBranch(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            init_method='normal_init'
        )

        self.reg_head = REGHead(
            feat_channels=256,
            num_anchors=self.num_anchors,
            num_regress=self.num_regress  # x, y, w, h
        )

        # self.loss ==> Max IoU Assigner with Focal Loss and Smooth L1 loss
        self.loss = IntegratedLoss(func=loss_func)

        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'resnet34':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet34(pretrained=self.pretrained)
            self.fpn_in_channels = [128, 256, 512]

        elif backbone == 'resnet50':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet50(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]

        elif backbone == 'resnet101':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet101(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]

        elif backbone == 'resnet152':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet101(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError

        del self.backbone.avgpool
        del self.backbone.fc

    def backbone_output(self, imgs):
        feature = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(imgs)))
        c2 = self.backbone.layer1(self.backbone.maxpool(feature))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c3, c4, c5]

    def forward(self, images, annots=None, image_names=None, test_conf=None):
        anchors_list, offsets_list = [], []
        original_anchors, num_level_anchors = self.anchor_generator(images)
        anchors_list.append(original_anchors)

        features = self.fpn(self.backbone_output(images))

        cls_score = torch.cat([self.cls_head(self.cls_branch(feature)) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(self.reg_branch(feature), with_deform=False)
                              for feature in features], dim=1)

        # get the predicted bboxes
        predicted_boxes = torch.cat(
            [self.box_coder.decode(anchors_list[-1][index], bbox_pred[index]).unsqueeze(0)
             for index in range(len(bbox_pred))], dim=0).detach()

        if self.training:
            # Max IoU Assigner with Focal Loss and Smooth L1 loss
            loss_cls, loss_reg = self.loss(cls_score,  # cls_score with all levels
                                           bbox_pred,  # bbox_pred with all levels
                                           anchors_list[-1],
                                           annots,
                                           image_names)

            return loss_cls, loss_reg

        else:  # for model eval()
            return self.decoder(images, anchors_list[-1], cls_score, bbox_pred,
                                thresh=cfg.score_thr, nms_thresh=cfg.nms_thr,
                                test_conf=test_conf)

    def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=None, nms_thresh=None, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        device_id = cls_score[0].get_device()
        anchor = anchors[0]  # (batch, A, 4) to (A, 4)
        pred = bbox_pred[0]  # (batch, A, 4) to (A, 4)
        bboxes = self.box_coder.decode(anchor, pred, mode='xywh')
        bboxes = bboxes[None]  # (A, 4) to (batch, A, 4)
        bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:  # negative class is the number of the class
            return [torch.zeros(1), torch.ones(1) * len(cfg.classes), torch.zeros(1, 8)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]

        # horizontal NMS with CUDA Extension
        # todo: ready to move gpu_nms in nms_wrapper.py
        dets = torch.cat([bboxes, scores], dim=2)[0, :, :]
        if torch.is_tensor(dets):
            dets = dets.cpu().detach().numpy()

        keep_idx = gpu_nms(dets, nms_thresh, device_id=device_id)
        nms_scores, nms_class = cls_score[0, keep_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, keep_idx, :],
            anchors[0, keep_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        """Set BN.eval(), BN is in the model's Backbone. """
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                # is only used to make the bn.running_mean and running_var not change in training phase.
                layer.eval()

                # freeze the bn.weight and bn.bias which are two learnable params in BN Layer.
                # layer.weight.requires_grad = False
                # layer.bias.requires_grad = False
