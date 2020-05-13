import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data.minibatch import get_minibatch
import utils.blob as blob_utils
from copy import deepcopy
import math
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
np.random.seed(666)
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes


class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, full_info_list, ratio_list, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)
        self.full_info_list = full_info_list # roidb_index, cls_list, image_id_list, roidb_index is useless.
        self.now_info_list = full_info_list
        self.ratio_list = ratio_list
        self.data_dict = {'ratio_index': self.full_info_list[:, 0], 'cls_ls': self.full_info_list[:, 1], 'img_ls': self.full_info_list[:, 2]}
        self.index_pd = pd.DataFrame.from_dict(self.data_dict)
        self.index_pd = self.index_pd.reset_index()

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple # this index is just the index of roidb, not the roidb_index
        # Get support roidb, support cls is same with query cls, and support image is different from query image.
        #query_cls = self.full_info_list[index, 1]
        #query_image = self.full_info_list[index, 2]
        query_cls = self.index_pd.loc[self.index_pd['index']==index, 'cls_ls'].tolist()[0]
        query_img = self.index_pd.loc[self.index_pd['index']==index, 'img_ls'].tolist()[0]
        all_cls = self.index_pd.loc[self.index_pd['img_ls']==query_img, 'cls_ls'].tolist()
        #support_blobs, support_valid = get_minibatch(support_db)
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db)
        #TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.
        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        if self._roidb[index]['need_crop']:
            self.crop_data(blobs, ratio)
            # Check bounding box
            entry = blobs['roidb'][0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]
        # Crop support data and get new support box in the support data
        support_way = 2 #2 #5 #2
        support_shot = 5 #5
        support_data_all = np.zeros((support_way * support_shot, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((support_way * support_shot, 4), dtype = np.float32)
        used_img_ls = [query_img]
        used_index_ls = [index]
        #used_cls_ls = [query_cls]
        used_cls_ls = list(set(all_cls))
        support_cls_ls = []
        mixup_i = 0

        for shot in range(support_shot):
            # Support image and box
            support_index = self.index_pd.loc[(self.index_pd['cls_ls'] == query_cls) & (~self.index_pd['img_ls'].isin(used_img_ls)) & (~self.index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
            support_cls = self.index_pd.loc[self.index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
            support_img = self.index_pd.loc[self.index_pd['index'] == support_index, 'img_ls'].tolist()[0]
            used_index_ls.append(support_index) 
            used_img_ls.append(support_img)

            support_db = [self._roidb[support_index]]
            support_data, support_box = self.crop_support(support_db)
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box[0]
            support_cls_ls.append(support_cls) #- 1)
            #assert support_cls - 1 >= 0
            mixup_i += 1

        if support_way == 1:
            pass
        else:
            for way in range(support_way-1):
                other_cls = self.index_pd.loc[(~self.index_pd['cls_ls'].isin(used_cls_ls)), 'cls_ls'].drop_duplicates().sample(random_state=index).tolist()[0]
                used_cls_ls.append(other_cls)
                for shot in range(support_shot):
                    # Support image and box

                    support_index = self.index_pd.loc[(self.index_pd['cls_ls'] == other_cls) & (~self.index_pd['img_ls'].isin(used_img_ls)) & (~self.index_pd['index'].isin(used_index_ls)), 'index'].sample(random_state=index).tolist()[0]
                     
                    support_cls = self.index_pd.loc[self.index_pd['index'] == support_index, 'cls_ls'].tolist()[0]
                    support_img = self.index_pd.loc[self.index_pd['index'] == support_index, 'img_ls'].tolist()[0]

                    used_index_ls.append(support_index) 
                    used_img_ls.append(support_img)
                    support_db = [self._roidb[support_index]]
                    support_data, support_box = self.crop_support(support_db)
                    support_data_all[mixup_i] = support_data
                    support_box_all[mixup_i] = support_box[0]
                    support_cls_ls.append(support_cls) #- 1)
                    #assert support_cls - 1 >= 0
                    mixup_i += 1

        blobs['support_data'] = support_data_all #final_support_data #support_blobs['data']
        blobs['roidb'][0]['support_boxes'] = support_box_all #support_blobs['roidb'][0]['boxes'] # only one box
        blobs['roidb'][0]['support_id'] = support_db[0]['id']
        #blobs['roidb'][0]['gt_classes'] = blobs['roidb'][0]['gt_classes'] #np.array([1] * (len(blobs['roidb'][0]['gt_classes'])))

        blobs['roidb'][0]['support_cls'] = support_cls_ls
        blobs['roidb'][0]['query_id'] = single_db[0]['id']
        blobs['roidb'][0]['target_cls'] = single_db[0]['target_cls']

        blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn
        return blobs

    def vis_image(self, im, bbox, im_name, output_dir):
        dpi = 300
        fig, ax = plt.subplots() 
        ax.imshow(im, aspect='equal') 
        plt.axis('off') 
        height, width, channels = im.shape 
        fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.margins(0,0)
        # Show box (off by default, box_alpha=0.0)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.5, alpha=1))
        output_name = os.path.basename(im_name)
        plt.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def crop_support(self, roidb):
        # Get support box
        '''
        data_height, data_width = map(int, blobs['im_info'][:2])
        im_scale = blobs['im_info'][-1]
        #print(blobs['roidb'])
        print(blobs['im_info'])
        print('support data:', blobs['data'].shape)
        all_box = (blobs['roidb'][0]['boxes'] * im_scale).astype(np.int16)
        '''
        img_path = roidb[0]['image']
        all_box = roidb[0]['boxes']
        all_cls = np.array(roidb[0]['gt_classes'])
        target_cls = roidb[0]['target_cls']

        target_idx = np.where(all_cls == target_cls)[0] 

        img = cv2.imread(img_path)
        if roidb[0]['flipped']:
            img = img[:, ::-1, :]
        img = img.astype(np.float32, copy=False)
        img -= cfg.PIXEL_MEANS
        img = img.transpose(2,0,1)
        data_height = int(img.shape[1])
        data_width = int(img.shape[2])
         
        all_box_num = all_box.shape[0]
        picked_box_id = np.random.choice(target_idx) #random.choice(range(all_box_num))
        #print(all_cls, target_cls, target_idx, picked_box_id)
        picked_box = all_box[picked_box_id,:][np.newaxis, :].astype(np.int16)
        #print('1', blobs['roidb'][0]['boxes'], '2', roidb[0]['boxes'], '3', picked_box) 
        '''
        original_img = cv2.imread(img_path)
        if roidb[0]['flipped']:
            original_img = original_img[:, ::-1, :]
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        self.vis_image(original_img, picked_box[0], img_path.split('/')[-1][:-4] + '_real.jpg', './test')
        '''
        '''
        picked_box = (picked_box / im_scale).astype(np.int16)
        data_height = int(data_height / im_scale)
        data_width = int(data_width / im_scale)
        blobs['data'] = cv2.resize(blobs['data'], (data_height, data_width))
        '''
        #print('3', picked_box)
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
        #self.vis_image(show_square, [new_x1, new_y1, new_x2, new_y2], img_path.split('/')[-1][:-4]+'_crop.jpg', './test')

        support_data = square
        support_box = np.array([[new_x1, new_y1, new_x2, new_y2]]).astype(np.float32)
        return support_data, support_box

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :,]
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            #print('min_x, max_x:', min_x, max_x)
            box_region = max_x - min_x + 1
            #print('box_region, size_crop:', box_region, size_crop)
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    print(x_s_min, x_s_max)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes


    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)

        if cfg.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, cfg.TRAIN.IMS_PER_BATCH)
            round_num_data = n * cfg.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            npr.shuffle(indices.reshape(-1, cfg.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
        else:
            rand_perm = npr.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list
