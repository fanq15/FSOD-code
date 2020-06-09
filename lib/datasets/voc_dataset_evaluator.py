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

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import shutil
import uuid

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from datasets.voc_eval import voc_eval
from utils.io import save_object
import json
import pandas as pd
logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=True,
    cleanup=True,
    use_matlab=False
):
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_files(json_dataset, all_boxes, salt)
    _do_python_eval(json_dataset, salt, output_dir)
    if use_matlab:
        _do_matlab_eval(json_dataset, salt, output_dir)
    if cleanup:
        for filename in filenames:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return None


def _write_voc_results_files(json_dataset, all_boxes, salt):
    filenames = []
    image_set_path = voc_info(json_dataset)['image_set_path']
    #print(image_set_path)
    assert os.path.exists(image_set_path), \
        'Image set path does not exist: {}'.format(image_set_path)
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()] ####################### need to change
    # Sanity check that order of images in json dataset matches order in the
    # image set
    '''
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = str(os.path.splitext(os.path.split(entry['image'])[1])[0])
        #print(index, image_index[i])
        assert str(index) == str(image_index[i])
    '''
    for cls_ind, cls in enumerate(json_dataset.classes): ############## need to change json_dataset.classes
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset,
                                                  salt).format(cls)
        filenames.append(filename)
        assert len(all_boxes[cls_ind]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if type(dets) == list:
                    assert len(dets) == 0, \
                        'dets should be numpy.ndarray or empty list'
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return filenames


def _get_voc_results_file_template(json_dataset, salt):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    devkit_path = info['devkit_path']
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    #return os.path.join(devkit_path, 'results', 'VOC' + year, 'Main', filename)
    return os.path.join('data/VOC' + year, 'Results', '{:s}.txt')


def _do_python_eval(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    #cachedir = os.path.join(devkit_path, 'annotations_cache')
    cachedir = os.path.join('data/VOC' + year, 'annotations_cache')
    aps_50 = []
    cls_ls = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load gt
    with open(image_set_path, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    # load annots
    recs = {}
    with open('./data/fsod/annotations/fsod_test.json', 'r') as f:
        now_dict = json.load(f)
    anno = pd.DataFrame.from_dict(now_dict['annotations'])
    category = pd.DataFrame.from_dict(now_dict['categories'])
    # we need image_id and category
    for i, imagename in enumerate(imagenames):
        image_now_ls = []
        image_id, category_id, ep = imagename.strip().split('_')
        category_name = category.loc[category['id'] == int(category_id), 'name'].tolist()[0]
        now_df = anno.loc[(anno['image_id'] == int(image_id)) & (anno['category_id'] == int(category_id)), :] #anno[anno['image_id'] == int(image_id)]
        obj_dict = now_df.to_dict('index')
        for key, value in obj_dict.items():
            value['name'] = category_name + '_' + ep  #category_name #category[category['id'] == value['category_id']]['name'].values[0]
            value['bbox'] = [value['bbox'][0], value['bbox'][1], value['bbox'][0] + value['bbox'][2], value['bbox'][1] + value['bbox'][3]]
            #print(value['name'])
            value['truncated'] = 0
            value['difficult'] = 0
            image_now_ls.append(value)
        recs[imagename] = image_now_ls
        if i % 100 == 0:
            logger.info(
                'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))

    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        rec, prec, ap = voc_eval(
            filename, anno_path, recs, imagenames, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps_50 += [ap]
        cls_ls += [cls]
        #logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_50_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    logger.info('Mean AP50 = {:.4f}'.format(np.mean(aps_50)))
    logger.info('~~~~~~~~')
    logger.info('AP50 Results:')
    for ap_id, ap in enumerate(aps_50):
        logger.info('AP50 for {} = {:.4f}'.format(cls_ls[ap_id], ap))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')

    aps_75 = []
    cls_ls = []
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(
            json_dataset, salt).format(cls)
        rec, prec, ap = voc_eval(
            filename, anno_path, recs, imagenames, cls, cachedir, ovthresh=0.75,
            use_07_metric=use_07_metric)
        aps_75 += [ap]
        cls_ls += [cls]
        #logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_75_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    logger.info('Mean AP75 = {:.4f}'.format(np.mean(aps_75)))
    logger.info('~~~~~~~~')
    logger.info('AP75 Results:')
    for ap_id, ap in enumerate(aps_75):
        logger.info('AP75 for {} = {:.4f}'.format(cls_ls[ap_id], ap))
    logger.info('~~~~~~~~')
    logger.info('Mean AP50 = {:.4f}'.format(np.mean(aps_50)))
    logger.info('Mean AP75 = {:.4f}'.format(np.mean(aps_75)))
    logger.info('~~~~~~~~')
    logger.info('')
    logger.info('----------------------------------------------------------')
    logger.info('Results computed with the **unofficial** Python eval code.')
    logger.info('Results should be very close to the official MATLAB code.')
    logger.info('Use `./tools/reval.py --matlab ...` for your paper.')
    logger.info('-- Thanks, The Management')
    logger.info('----------------------------------------------------------')


def _do_matlab_eval(json_dataset, salt, output_dir='output'):
    import subprocess
    logger.info('-----------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code.')
    logger.info('-----------------------------------------------------')
    info = voc_info(json_dataset)
    path = os.path.join(
        cfg.ROOT_DIR, 'lib', 'datasets', 'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
       .format(info['devkit_path'], 'comp4' + salt, info['image_set'],
               output_dir)
    logger.info('Running:\n{}'.format(cmd))
    subprocess.call(cmd, shell=True)


def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = 'test' #'val' #json_dataset.name[9:]
    devkit_path = None #DATASETS[json_dataset.name][DEVKIT_DIR]
    #assert os.path.exists(devkit_path), \
    #    'Devkit directory {} not found'.format(devkit_path)
    #anno_path = os.path.join(
    # 'data/VOC' + year, 'Annotations', '{:s}.xml')   
    #    devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    #image_set_path = os.path.join(
    #    'data/VOC' + year, 'ImageSets', 'Main', 'new_val.txt')
    #    devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
    anno_path = None
    image_set_path = './data/fsod/new_val.txt'
    return dict(
        year=year,
        image_set=image_set,
        devkit_path=devkit_path,
        anno_path=anno_path,
        image_set_path=image_set_path)
