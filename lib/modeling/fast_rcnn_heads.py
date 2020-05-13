import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.patch_relation = True
        self.local_correlation = True
        self.global_relation = True

        if self.patch_relation:
            self.conv_1 = nn.Conv2d(dim_in*2, int(dim_in/4), 1, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(int(dim_in/4), int(dim_in/4), 3, padding=0, bias=False)
            self.conv_3 = nn.Conv2d(int(dim_in/4), dim_in, 1, padding=0, bias=False)
            self.bbox_pred_pr = nn.Linear(dim_in, 4 * 2)
            self.cls_score_pr = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)
 
        if self.local_correlation:
            self.conv_cor = nn.Conv2d(dim_in, dim_in, 1, padding=0, bias=False)
            #self.bbox_pred_cor = nn.Linear(dim_in, 4 * 2)
            self.cls_score_cor = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)

        if self.global_relation:
            self.fc_1 = nn.Linear(dim_in * 2, dim_in)
            self.fc_2 = nn.Linear(dim_in, dim_in)
            self.cls_score_fc = nn.Linear(dim_in, 2) #nn.Linear(dim_in, 2)

        self.avgpool = nn.AvgPool2d(kernel_size=3,stride=1)
        self.avgpool_fc = nn.AvgPool2d(7)

        self._init_weights()

    def _init_weights(self):

        if self.patch_relation:
            init.normal_(self.conv_1.weight, std=0.01)
            init.normal_(self.conv_2.weight, std=0.01)
            init.normal_(self.conv_3.weight, std=0.01)
            init.normal_(self.cls_score_pr.weight, std=0.01)
            init.constant_(self.cls_score_pr.bias, 0)
            init.normal_(self.bbox_pred_pr.weight, std=0.001)
            init.constant_(self.bbox_pred_pr.bias, 0)

        if self.local_correlation:
            init.normal_(self.conv_cor.weight, std=0.01)
            init.normal_(self.cls_score_cor.weight, std=0.01)
            init.constant_(self.cls_score_cor.bias, 0)

        if self.global_relation:
            init.normal_(self.fc_1.weight, std=0.01)
            init.constant_(self.fc_1.bias, 0)
            init.normal_(self.fc_2.weight, std=0.01)
            init.constant_(self.fc_2.bias, 0)
            init.normal_(self.cls_score_fc.weight, std=0.01)
            init.constant_(self.cls_score_fc.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'conv_1.weight': 'conv_1_w',
            'conv_2.weight': 'conv_2_w',
            'conv_3.weight': 'conv_3_w',
            'cls_score_pr.weight': 'cls_score_pr_w',
            'cls_score_pr.bias': 'cls_score_pr_b',
            'bbox_pred_pr.weight': 'bbox_pred_pr_w',
            'bbox_pred_pr.bias': 'bbox_pred_pr_b',

            'conv_cor.weight': 'conv_cor_w',
            'cls_score_cor.weight': 'cls_score_cor_w',
            'cls_score_cor.bias': 'cls_score_cor_b',

            'fc_1.weight': 'fc_1_w',
            'fc_1.bias': 'fc_1_b',
            'fc_2.weight': 'fc_2_w',
            'fc_2.bias': 'fc_2_b',
            'cls_score_fc.weight': 'cls_score_fc_w',
            'cls_score_fc.bias': 'cls_score_fc_b'#,
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x_query, x_support): #, way, shot):
        if self.global_relation:
            x_query_fc = self.avgpool_fc(x_query).squeeze(3).squeeze(2)
        if self.local_correlation:
            x_query_cor = self.conv_cor(x_query)
        support = x_support.mean(0, True)
        # fc
        if self.global_relation:
            support_fc = self.avgpool_fc(support).squeeze(3).squeeze(2).expand_as(x_query_fc)
            cat_fc = torch.cat((x_query_fc, support_fc), 1)
            out_fc = F.relu(self.fc_1(cat_fc), inplace=True)
            out_fc = F.relu(self.fc_2(out_fc), inplace=True)
            cls_score_fc = self.cls_score_fc(out_fc)

        # correlation
        if self.local_correlation:
            support_cor = self.conv_cor(support)
            x_cor = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=2048), inplace=True).squeeze(3).squeeze(2)
            cls_score_cor = self.cls_score_cor(x_cor)

        # relation
        if self.patch_relation:
            support_relation = support.expand_as(x_query)
            x = torch.cat((x_query, support_relation), 1)
            x = F.relu(self.conv_1(x), inplace=True) # 5x5
            x = self.avgpool(x)
            x = F.relu(self.conv_2(x), inplace=True) # 3x3
            x = F.relu(self.conv_3(x), inplace=True) # 3x3
            x = self.avgpool(x) # 1x1
            x = x.squeeze(3).squeeze(2)
            cls_score_pr = self.cls_score_pr(x)

        bbox_pred_all = self.bbox_pred_pr(x)
        # final result
        cls_score_all = cls_score_pr + cls_score_cor + cls_score_fc
        if not self.training:
            cls_score_all = F.softmax(cls_score_all, dim=1)

        return cls_score_all, bbox_pred_all

def fast_rcnn_losses_ohem(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    loss_cls = F.cross_entropy(cls_score, rois_label, reduction='none', ignore_index=-1)

    sorted, idx = torch.sort(loss_cls, descending=True)
    keep_num = min((rois_label == 1).nonzero().shape[0] * 4, cls_score.shape[0])
    keep_idx = idx[:keep_num]
    loss_cls = loss_cls[keep_idx]
    loss_cls = loss_cls.sum() / keep_num

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    return loss_cls, loss_bbox, accuracy_cls

def fast_rcnn_losses_original(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    loss_cls = F.cross_entropy(cls_score, rois_label)

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)

    return loss_cls, loss_bbox, accuracy_cls

def fast_rcnn_losses(cls_score_ori, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score_ori.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    cls_score_softmax = torch.nn.functional.softmax(cls_score_ori, dim=1)
    fg_inds = (rois_label == 1).nonzero().squeeze(-1)
    bg_inds = (rois_label == 0).nonzero().squeeze(-1)
    bg_cls_score_softmax = cls_score_softmax[bg_inds, :]

    bg_num_0 = max(1, min(fg_inds.shape[0] * 2, int(rois_label.shape[0] * 0.25)))
    bg_num_1 = max(1, min(fg_inds.shape[0], bg_num_0))

    sorted, sorted_bg_inds = torch.sort(bg_cls_score_softmax[:, 1], descending=True)

    real_bg_inds = bg_inds[sorted_bg_inds]
    real_bg_topk_inds_0 = real_bg_inds[real_bg_inds < int(rois_label.shape[0] * 0.5)][:bg_num_0]
    real_bg_topk_inds_1 = real_bg_inds[real_bg_inds >= int(rois_label.shape[0] * 0.5)][:bg_num_1]

    topk_inds = torch.cat([fg_inds, real_bg_topk_inds_0, real_bg_topk_inds_1], dim=0)
    loss_cls = F.cross_entropy(cls_score_ori[topk_inds], rois_label[topk_inds])

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score_ori.max(dim=1)[1].type_as(rois_label)
    accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    return loss_cls, loss_bbox, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
