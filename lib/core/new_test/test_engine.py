# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer
import pandas as pd
import math
np.random.seed(666)
logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        raise NotImplementedError
        # child_func = generate_rpn_on_range
        # parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb(gt=True))
        print('total images: ', num_images)
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images, index_pd = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)

    img_ls = []
    for item in roidb:
        img_ls.append(item['image'])
    num_annotations = len(roidb)
    num_images = len(list(set(img_ls)))
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)

    timers = defaultdict(Timer)

    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        # Get support box
        index = entry['index']
        query_cls = index_pd.loc[index_pd['index']==index, 'cls_ls'].tolist()[0]
        query_img = index_pd.loc[index_pd['index']==index, 'img_ls'].tolist()[0]
        all_cls = index_pd.loc[index_pd['img_ls']==query_img, 'cls_ls'].tolist()

        support_way = 1
        support_shot = 1
        support_data_all = np.zeros((support_way * support_shot, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((support_way * support_shot, 4), dtype = np.float32)
        used_img_ls = [query_img]
        used_index_ls = [index]
        used_cls_ls = list(set(all_cls))
        support_cls_ls = []

        mixup_i = 0

        for shot in range(support_shot):
            # Support image and box
            support_index = index_pd.loc[(index_pd['cls_ls'] == query_cls) & (~index_pd['img_ls'].isin(used_img_ls)) & (~index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
            support_cls = index_pd.loc[index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
            support_img = index_pd.loc[index_pd['index'] == support_index, 'img_ls'].tolist()[0]
            used_index_ls.append(support_index) 
            used_img_ls.append(support_img)

            support_data, support_box = crop_support(entry)
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box[0]
            support_cls_ls.append(support_cls) #- 1)
            #assert support_cls - 1 >= 0
            mixup_i += 1

        if support_way == 1:
            pass
        else:
            for way in range(support_way-1):
                other_cls = index_pd.loc[(~index_pd['cls_ls'].isin(used_cls_ls)), 'cls_ls'].drop_duplicates().sample(random_state=index).tolist()[0]
                used_cls_ls.append(other_cls)
                for shot in range(support_shot):
                    # Support image and box

                    support_index = index_pd.loc[(index_pd['cls_ls'] == other_cls) & (~index_pd['img_ls'].isin(used_img_ls)) & (~index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
                     
                    support_cls = index_pd.loc[index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
                    support_img = index_pd.loc[index_pd['index'] == support_index, 'img_ls'].tolist()[0]

                    used_index_ls.append(support_index) 
                    used_img_ls.append(support_img)
                    support_data, support_box = crop_support(entry)
                    support_data_all[mixup_i] = support_data
                    support_box_all[mixup_i] = support_box[0]
                    support_cls_ls.append(support_cls) #- 1)
                    #assert support_cls - 1 >= 0
                    mixup_i += 1

        save_path = './vis'
        im = cv2.imread(entry['image'])
        cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, support_data_all, support_box_all, support_cls_ls, support_shot, save_path, box_proposals, timers)
        real_index = entry['real_index']
        cls_boxes_i = cls_boxes_i[1]
        for cls in support_cls_ls:
            extend_support_results(real_index, all_boxes, cls_boxes_i[cls_boxes_i[:,5] == cls][:, :5], cls)

        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_annotations - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_annotations, det_time, misc_time, eta
                )
            )

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                segms=cls_segms_i,
                keypoints=cls_keyps_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes, all_segms, all_keyps

def crop_support(entry):
    # Get support box
    img_path = entry['image']
    all_box = entry['boxes']
    all_cls = np.array(entry['gt_classes'])
    target_cls = entry['target_cls']

    target_idx = np.where(all_cls == target_cls)[0] 

    img = cv2.imread(img_path)
    if entry['flipped']:
        img = img[:, ::-1, :]
    img = img.astype(np.float32, copy=False)
    img -= cfg.PIXEL_MEANS
    img = img.transpose(2,0,1)
    data_height = int(img.shape[1])
    data_width = int(img.shape[2])
     
    all_box_num = all_box.shape[0]
    picked_box_id = np.random.choice(target_idx) #random.choice(range(all_box_num))
    picked_box = all_box[picked_box_id,:][np.newaxis, :].astype(np.int16)
    '''
    original_img = cv2.imread(img_path)
    if entry['flipped']:
        original_img = original_img[:, ::-1, :]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    vis_image(original_img, picked_box[0], img_path.split('/')[-1][:-4] + '_real.jpg', './test')
    '''
    x1 = picked_box[0][0]
    y1 = picked_box[0][1]
    x2 = picked_box[0][2] 
    y2 = picked_box[0][3]

    width = x2 - x1
    height = y2 - y1
    context_pixel = 16 #int(16 * im_scale)
    
    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height

    target_size = (320, 320) #(384, 384)   

    if width >= height:
        crop_x1 = x1 - context_pixel
        crop_x2 = x2 + context_pixel
   
        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + context_pixel
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width
            
        short_size = height
        long_size = crop_x2 - crop_x1
        y_center = int((y2+y1) / 2) #math.ceil((y2 + y1) / 2)
        crop_y1 = int(y_center - (long_size / 2)) #int(y_center - math.ceil(long_size / 2))
        crop_y2 = int(y_center + (long_size / 2)) #int(y_center + math.floor(long_size / 2))
        
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height

        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.float32)
        delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]

        square[:, square_y1:square_y2, :] = crop_box

        #show_square = np.zeros((crop_long_size, crop_long_size, 3))#, dtype=np.int16)
        #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        #show_square[square_y1:square_y2, :, :] = show_crop_box
        #show_square = show_square.astype(np.int16)
    else:
        crop_y1 = y1 - context_pixel
        crop_y2 = y2 + context_pixel
   
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + context_pixel
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
            
        short_size = width
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2) #math.ceil((x2 + x1) / 2)
        crop_x1 = int(x_center - (long_size / 2)) #int(x_center - math.ceil(long_size / 2))
        crop_x2 = int(x_center + (long_size / 2)) #int(x_center + math.floor(long_size / 2))

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width


        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.float32)
        delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

        #show_square = np.zeros((crop_long_size, crop_long_size, 3)) #, dtype=np.int16)
        #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        #show_square[:, square_x1:square_x2, :] = show_crop_box
        #show_square = show_square.astype(np.int16)

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / long_size
    square = square.transpose(1,2,0)
    square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    square = square.transpose(2,0,1)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    # For test
    #show_square = cv2.resize(show_square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    #vis_image(show_square, [new_x1, new_y1, new_x2, new_y2], img_path.split('/')[-1][:-4]+'_crop.jpg', './test')

    support_data = square
    support_box = np.array([[new_x1, new_y1, new_x2, new_y2]]).astype(np.float32)
    return support_data, support_box


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb(gt=True)

    for item in roidb:
        all_cls = item['gt_classes']
        target_cls = item['target_cls']
        target_idx = np.where(all_cls == target_cls)[0] 
        item['boxes'] = item['boxes'][target_idx]
        item['gt_classes'] = item['gt_classes'][target_idx]

    print('testing annotation number: ', len(roidb))
    roidb_img = []
    roidb_cls = []
    roidb_index = []
    for item in roidb:
        roidb_img.append(item['image'])
        roidb_cls.append(item['target_cls'])
        roidb_index.append(item['index'])
    data_dict = {'img_ls': roidb_img, 'cls_ls': roidb_cls, 'index': roidb_index}
    index_pd = pd.DataFrame.from_dict(data_dict)
   
    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
        for item in roidb:
            item['real_index'] -= start
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images, index_pd


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps



def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]

def extend_support_results(index, all_res, im_res, cls_idx):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    #for cls_idx in range(1, len(im_res)):
    all_res[cls_idx][index] = im_res #[1]
