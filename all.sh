SAVE_DIR=fsod
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net_step.py --save_dir $SAVE_DIR --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --bs 4 --iter_size 2 --nw 4 --load_detectron data/pretrained_model/model_final.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/test_net.py --multi-gpu-testing --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --load_ckpt Outputs/$SAVE_DIR/ckpt/model_step59999.pth

