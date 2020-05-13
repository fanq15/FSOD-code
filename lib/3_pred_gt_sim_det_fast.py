import pickle
from scipy.spatial import distance
import numpy as np
import random
import time
from copy import deepcopy
import os
random.seed(666)
from utils.cython_bbox import bbox_overlaps
import utils.boxes as box_utils
import matplotlib.pyplot as plt
import cv2
#cimport cython
#cimport numpy as np

# setting
#num_way = 1
#thres = 0.7
pkl_path = '../log/gt.pkl'

def vis_image(im, bbox, im_name, output_dir):
    dpi = 500
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)
    # show box (off by default, box_alpha=0.0)
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1],
                      fill=False, edgecolor='r',
                      linewidth=0.5, alpha=1))
    output_name = os.path.basename(im_name)
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def get_pkl_ls(pkl_path):
    pkl_ls = []
    with open(pkl_path, 'rb') as f:
        while 1:
            try:
                a = pickle.load(f)
                pkl_ls.append(a)
            except:
                break
    return pkl_ls

gt_pkl_path = '../log/gt.pkl'
real_pkl_ls = get_pkl_ls(gt_pkl_path)

gt_img_info_path = '../log/gt_img_info.pkl'
gt_info_ls = get_pkl_ls(gt_img_info_path)

pred_pkl_path = '../log/pred_box_50.pkl'
pred_pkl_ls = get_pkl_ls(pred_pkl_path)

#pred_img_info_path = '../log/pred_img_info.pkl'
#pred_info_ls = get_pkl_ls(pred_img_info_path)

def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall list
    :param prec: precision list
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.

        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0    
            else:
                p = np.max(prec[rec >= t])   
            ap = ap + p / 11.    # 11-recall-point based AP
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
       
        i = np.where(mrec[1:] != mrec[:-1])[0]

        
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_box_cls_ls(pkl_ls):
    '''
    box_cls_ls: 
    '''
    pkl_id_ls = range(len(pkl_ls))
    box_cls_ls = [[] for i in range(0,21)] # box_cls_ls[0] is background and its number is 0 when use gt box 
    cls_min_num = 99999999999
    for img_id in pkl_id_ls:
        img = pkl_ls[img_id]
        roidb = img['roidb']
        for box_id in range(roidb.shape[0]):
            box_name = str(img_id) + '-' + str(box_id)
            roidb_i = roidb[box_id]
            roidb_i_class = roidb_i[0]
            #print(roidb_i_class)
            box_cls_ls[int(roidb_i_class)].append(box_name)
    for item_id in range(len(box_cls_ls)): # shuffle list and get cls_min_num
        cls_num = len(box_cls_ls[item_id])
        if cls_num < cls_min_num and cls_num != 0:
            cls_min_num = cls_num
        #print(cls_num, cls_min_num)
        #print(box_cls_ls[item_id])
        random.shuffle(box_cls_ls[item_id])
        #print(box_cls_ls[item_id])
        #print(len(box_cls_ls[item_id]))

    for item_id in range(len(box_cls_ls)): # get list[:cls_min_num]
        box_cls_ls[item_id] = box_cls_ls[item_id][:cls_min_num]
        #print(len(box_cls_ls[item_id]))
    return box_cls_ls 
 


def calculate_sim_ap(pkl_ls, gt_info_ls, pred_ls, thres): 
    ap_ls = []
    total_precision = []
    total_recall = []
    count_all = 0
    tp_all = 0
    gt_all = 0
    ap_all_ls = []
    support_box_cls_ls = get_box_cls_ls(pkl_ls)    #print(len(support_box_cls_ls[0]))

    # for pred
    # fore pred_ls
    box_num = 0
    #print(pred_ls)
    for img_id, img in enumerate(pred_ls):
        #print(img_id, img)
        box_feat = np.squeeze(img['box_feat'], axis=(2,3))
        box_num += box_feat.shape[0]
        feat_len = box_feat.shape[1]

    print('pred_box_num, feat_len:', box_num, feat_len)
    #assert False

    query_roidb_ls = np.zeros((box_num, 5), dtype = np.float32) # init query_roidb_ls for the try-except
    query_box_feat_ls = np.zeros((box_num, feat_len), dtype = np.float32)
    pred_img_ls = []
    for img_id, img in enumerate(pred_ls):
        #print(str(img_id), img['box_feat'].shape)
        box_feat = np.squeeze(img['box_feat'], axis=(2,3))
        roidb = img['roidb']
        #print(roidb)
        current_box_num = img['box_feat'].shape[0]
        box_start = len(pred_img_ls)
        query_roidb_ls[box_start:box_start + current_box_num, :-1] = roidb[:, 1:]
        query_box_feat_ls[box_start:box_start + current_box_num, :] = box_feat
        
        pred_img_ls.extend([img_id]*current_box_num) # get img id of every box
    pred_img_ls = np.array(pred_img_ls)
    '''
    # for pred_info_ls
    pred_id_ls = []
    for img_id, img in enumerate(pred_info_ls):
        pred_id_ls.append(img['entry']['id'])
        #print('pred_info:', img['entry']['id'])
    '''
    # for gt
    # for pkl_ls
    gt_img_ls = []
    gt_cls_ls = []
    gt_box_num = 0
    #print(pred_ls)
    for img_id, img in enumerate(pkl_ls):
        #print(img_id, img)
        box_feat = np.squeeze(img['box_feat'], axis=(2,3))
        gt_box_num += box_feat.shape[0]
        feat_len = box_feat.shape[1]

    print('gt_box_num, feat_len:', gt_box_num, feat_len)

    gt_box_ls = np.zeros((gt_box_num, 4), dtype = np.float32)
    for img_id, img in enumerate(pkl_ls):
        roidb = img['roidb']
        #print(roidb.shape)
        current_box_num = roidb.shape[0]
        box_start = len(gt_img_ls)
        gt_box_ls[box_start:box_start + current_box_num, :] = roidb[:, 1:]
        gt_img_ls.extend([img_id]*current_box_num) # get img id of every box
        gt_cls_ls.extend(roidb[:, 0].astype(np.int16).tolist())

    # for gt_info_ls
    gt_id_ls = []
    gt_path_ls = []
    for img_id, img in enumerate(gt_info_ls):
        gt_id_ls.append(img['entry']['id'])
        gt_path_ls.append(img['entry']['image'])
        #print('gt_info:', img['entry']['id'])

    #print(pred_img_ls)
    for support_id in range(len(support_box_cls_ls[-1])): # pick -1 to avoid 0 bg num
        support_roidb_ls = np.array([])
        support_box_feat_ls = np.array([])
        support_box_ls = []
        support_img_ls = []
        support_id_ls = []

        for cls_id in range(len(support_box_cls_ls)):
            if len(support_box_cls_ls[cls_id]) == 0: # for 0 bg number
                # assert False # it is bug when use predicted boxes
                continue
            else:
                support_item = support_box_cls_ls[cls_id][support_id]
                support_img_id = int(support_item.split('-')[0])
                support_box_id = int(support_item.split('-')[-1])
                support_img_ls.append(support_img_id)
                support_box_ls.append(support_box_id)
            #print(support_box_id)
            support_img = pkl_ls[int(support_img_id)]
            support_box_feat = np.squeeze(support_img['box_feat'], axis=(2,3))[int(support_box_id)][np.newaxis, :]
            support_roidb = support_img['roidb'][int(support_box_id)][np.newaxis, :]
            #print(support_box_feat.shape)
            if support_roidb_ls.shape[0] != 0:
                support_roidb_ls = np.concatenate((support_roidb_ls, support_roidb), axis=0)
                support_box_feat_ls = np.concatenate((support_box_feat_ls, support_box_feat), axis=0)
            else:
                support_roidb_ls = support_roidb
                support_box_feat_ls = support_box_feat
            #print(support_roidb_ls.shape)
            support_id_ls.append(str(cls_id) + '_' + support_item)
        support_img_ls = np.array(support_img_ls)
        start = time.time()
        #print(support_img_ls.shape)
        #print(query_roidb_ls.shape[0])

        #print(query_box_feat_ls.shape)
        #print(support_box_feat_ls.shape)
        #print(pred_img_ls.shape)
        #print('support_img_ls:', support_img_ls)
        #print(np.in1d(pred_img_ls, support_img_ls).sum())
        #continue
        #print(np.in1d(pred_img_ls, support_img_ls).shape)
        query_box_feat_ls_now = deepcopy(query_box_feat_ls)
        query_box_feat_ls_now = query_box_feat_ls_now[~np.in1d(pred_img_ls, support_img_ls)]
        sim_matrix = 1 - distance.cdist(query_box_feat_ls_now, support_box_feat_ls, 'cosine')
        sim_matrix_max = np.max(sim_matrix, axis=1)
        pred_box_ls = deepcopy(query_roidb_ls)
        pred_box_ls = pred_box_ls[~np.in1d(pred_img_ls, support_img_ls)]
        pred_box_ls[:, -1] = sim_matrix_max
        pred_img_ls_now = pred_img_ls[~np.in1d(pred_img_ls, support_img_ls)]

        sim_matrix_bg = np.where(sim_matrix_max <= thres) # if sim_matrix_max <= thres, it is background
        sim_matrix_cls = np.argmax(sim_matrix, axis=1) + 1 # wrong: when use gt box, it needs to +1, when use predicted boxes, remove +1.
        sim_matrix_cls[sim_matrix_bg] = 0

        pred_cls_ls = deepcopy(sim_matrix_cls)
        #print(time.time() - start)
        
        #pred_box_num = sim_matrix_cls.shape[0]
        #tp = np.zeros(pred_box_num)
        #fp = np.zeros(pred_box_num)
        
        current_gt_cls_ls = np.array(gt_cls_ls)[~np.in1d(gt_img_ls, support_img_ls)]
        current_gt_img_ls = np.array(gt_img_ls)[~np.in1d(gt_img_ls, support_img_ls)]
        current_gt_box_ls = deepcopy(gt_box_ls)[~np.in1d(gt_img_ls, support_img_ls)]

        cls_unique = np.unique(current_gt_cls_ls)
        img_unique = np.unique(current_gt_img_ls)

        ap_ls = []
        ovthresh = 0.5
        for cls in cls_unique:
            '''
            nd = (pred_cls_ls==cls).sum()
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            print('nd:', nd)
            for img_id, img in enumerate(pred_img_ls.tolist()):
                # for pred

                pred_cls = deepcopy(pred_cls_ls)[img_id]
                #print(pred_cls, cls)
                if pred_cls != cls:
                    continue
            '''
            tp = []
            fp = []
            npos = (current_gt_cls_ls==cls).sum()
            #print(npos)
            total_pred_box = 0
            total_pred_box_after_nms = 0
            total_gt_box = 0
            total_img = 0
            start = time.time()
            confidence_ls = []
            for img in img_unique:
                #assert pred_id_ls[img] == gt_id_ls[img]
                pred_box = deepcopy(pred_box_ls)[pred_img_ls_now==img, :]
                pred_cls = deepcopy(pred_cls_ls)[pred_img_ls_now==img]

                pred_box = pred_box[pred_cls==cls, :]
                keep = box_utils.nms(pred_box, 0.3)
                nms_box = pred_box[keep, :]
            
                
                # for gt
                gt_box = current_gt_box_ls[current_gt_img_ls==img, :]
                gt_cls = current_gt_cls_ls[current_gt_img_ls==img]
                gt_box = gt_box[gt_cls==cls, :]
                '''
                if nms_box.shape[0] == 0:
                    tp.extend([0.] * gt_box.shape[0])
                    fp.extend([1.] * gt_box.shape[0])
                    continue
                if gt_box.shape[0] == 0:
                    tp.extend([0.] * nms_box.shape[0])
                    fp.extend([1.] * nms_box.shape[0])
                    continue
                '''
                confidence = nms_box[:, -1]
                sorted_ind = np.argsort(-confidence)
                #print(confidence, sorted_ind)
                nms_box = nms_box[sorted_ind, :-1]

                # sort tp and fp according confidence
                confidence = confidence[sorted_ind]
                confidence_ls.extend(confidence.tolist())

                det_flag = [False] * gt_box.shape[0]
                BBGT = gt_box.astype(float)
                total_pred_box += pred_box.shape[0]
                total_pred_box_after_nms += nms_box.shape[0]
                total_gt_box += gt_box.shape[0]
                if gt_box.shape[0] != 0:
                    total_img += 1
                for box_i in nms_box:
                    bb = box_i.astype(float)

                    ovmax = -np.inf
                    #start1 = time.time()
                    if BBGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(BBGT[:, 0], bb[0])
                        iymin = np.maximum(BBGT[:, 1], bb[1])
                        ixmax = np.minimum(BBGT[:, 2], bb[2])
                        iymax = np.minimum(BBGT[:, 3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    #print(time.time() - start1)
                    if ovmax > ovthresh:
                        if not det_flag[jmax]:
                            tp.append(1.)
                            fp.append(0.)
                            
                            #tp[img_id] = 1.
                            det_flag[jmax] = 1
                        else:
                            tp.append(0.)
                            fp.append(1.)
                            #fp[img_id] = 1.
                        
                    else:
                        tp.append(0.)
                        fp.append(1.)
                        #fp[img_id] = 1.
            tp = np.array(tp)
            fp = np.array(fp)
            confidence_ls = np.array(confidence_ls)
            sorted_ind = np.argsort(-confidence_ls)
            tp = tp[sorted_ind]
            fp = fp[sorted_ind]
            img_id = int(support_img_ls[cls-1])
            #print(img_id, gt_path_ls[img_id])
            
            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            #print(fp[-1], tp[-1])
            #print('total pred box:', total_pred_box, ' total_pred_box_after_nms:', total_pred_box_after_nms, ' total_gt_box:', total_gt_box, 'total_image:', total_img)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            current_ap = voc_ap(rec, prec)
            #print('time:', time.time() - start)
            #print('cls, rec, prec, ap', cls, rec[-1], prec[-1], current_ap)

            # vis
            im_path = gt_path_ls[img_id]
            im_name = im_path.split('/')[-1]
            im_dir = os.path.join('./log/figure', str(thres), 'class_' + str(cls))
            output_name = '%.4f' % round(current_ap, 4) + '_' + str(im_name)
            if not os.path.exists(im_dir):
                os.makedirs(im_dir)
            im = cv2.imread(im_path)
            current_support_box = support_roidb_ls[cls-1][1:]
            assert cls == int(support_roidb_ls[cls-1][0])
            #print(current_support_box)
            #print(round(current_ap, 4))
            vis_image(
                im[:, :, ::-1],
                current_support_box, 
                output_name,
                im_dir, #os.path.join(output_dir, 'vis'),
            )
            #assert False

            #print(support_id_ls)
            ap_ls.append(current_ap)
            if cls == 1:
                ap_save = str(thres) + ',' +  str(support_id) + ',' + str(current_ap)
                ap_save = ap_save + ',' + str(current_ap) + ',' + support_id_ls[cls_id-1]
            else:
                ap_save = ap_save + ',' + str(current_ap) + ',' + support_id_ls[cls_id-1]
        ap = sum(ap_ls) / float(len(ap_ls)) # remove bg when use gt boxes, when use predicted boxes, remove -1
        ap_all_ls.append(ap)
        print('support_id:', support_id, '  threshold:', thres, '  ap:',  ap)
        root_path = './log'
        support_ap_path = os.path.join(root_path, 'support_ap.csv')
        all_ap_path = os.path.join(root_path, 'all_ap.csv')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        #if os.path.exists(support_ap_path):
        #    os.remove(support_ap_path)
        #if os.path.exists(all_ap_path):
        #    os.remove(all_ap_path)
        with open(support_ap_path, 'a') as f:
            f.write(str(thres) + ',' +  str(support_id) + ',' + str(ap) + '\n')
        with open(all_ap_path, 'a') as f1:
            f1.write(ap_save + '\n')
        
    return ap_all_ls

#num_way_ls = [20,30,40,50,60,70,80,90,100,110,120]
thres_ls = np.arange(0.3, 0.6, 0.05)
#iter_num = 5

total_iter = 1
head = 'id, threshold, mAP, std'+'\n'

root_path = './log'
result_det_ap_fast_path = os.path.join(root_path, 'result_det_ap_pred.csv')

if not os.path.exists(root_path):
    os.makedirs(root_path)
#if os.path.exists(result_det_ap_fast_path):
#    os.remove(result_det_ap_fast_path)

with open(result_det_ap_fast_path, 'a') as f1:
    f1.write(head)

for thres in thres_ls:
    thres = round(thres, 2)
    #for iter_id in range(iter_num):
    ap_all = np.array(calculate_sim_ap(real_pkl_ls, gt_info_ls, pred_pkl_ls, thres))
    mAP = np.mean(ap_all)
    std = np.std(ap_all, ddof=1)
    print('id:', total_iter, '  threshold:', thres, '  mAP:', mAP, ' std:', std)
    result = str(total_iter) + ',' + str(thres) + ',' + str(mAP) + ',' + str(std) + '\n'
    total_iter += 1

    with open(result_det_ap_fast_path, 'a') as f1:
        f1.write(result)











