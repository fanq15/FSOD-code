# A Pytorch Implementation of FSOD

## NEWS!

**Detectron2 based FSOD implementation is released in [FewX](https://github.com/fanq15/FewX)! It can reach 12.0 AP on MS COCO (10-shot)!**

## Getting Started
Clone the repo:

```
git clone https://github.com/fanq15/FSOD-code.git
```

### Requirements

Tested under python3.

- python packages
  - pytorch==0.4.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml==3.12
  - packaging
  - pandas
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 9.0 are required. (Do not use other versions)
- **NOTICE**: different versions of Pytorch package have different memory usages.

### Compilation

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```

If your are using Volta GPUs, uncomment this [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/tree/master/lib/make.sh#L15) in `lib/mask.sh` and remember to postpend a backslash at the line above. `CUDA_PATH` defaults to `/usr/loca/cuda`. If you want to use a CUDA library on different path, change this [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/tree/master/lib/make.sh#L3) accordingly.

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align. (Actually gpu nms is never used ...)

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compile the code.**

### Data Preparation

Please add `data` in the `fsod` directory and the structure is :

  ```
  YOUR_PATH
      └── fsod
            ├── code files
            └── data
                  ├──── fsod
                  |       ├── annotations
                  │       │       ├── fsod_train.json
                  │       │       └── fsod_test.json
                  │       └── images
                  │             ├── part_1
                  │             └── part_2
                  │ 
                  └──── pretrained_model
                          └── model_final.pkl (from detectron model zoo: End-to-End Faster & Mask R-CNN Baselines R-50-C4 Faster 2x model)
  ```  
You can download the model_final.pkl from here: [Model link](https://dl.fbaipublicfiles.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)

### Training and evaluation

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net_step.py --save_dir fsod_save_dir --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --bs 4 --iter_size 2 --nw 4 --load_detectron data/pretrained_model/model_final.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/test_net.py --multi-gpu-testing --dataset fsod --cfg configs/fsod/voc_e2e_faster_rcnn_R-50-C4_1x_old_1.yaml --load_ckpt Outputs/fsod_save_dir/ckpt/model_step59999.pth
```

The default setting is on 4 GPUs. If you want to change GPU numbers, please change `bs` in the training script and make sure `bs=#GPU`. If you want to use the default training setting on 4 GPUs, you can directly run `sh all.sh` to train and evaluate the model. 

### Others

Please note that this repository is only for episode training and evaluation on FSOD dataset. Other expriments on different datasets are in different evaluation settings and I will gradually merge them. Currently, the code only supports 1 image on per GPU for both training and evaluation.

This repository is originally built on [roytseng-tw/Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). You can reference it for more implementation details. 

## Citation

  If you use this dataset in your research, please cite this [paper](https://arxiv.org/pdf/1908.01998v1.pdf).

  ```
  @inproceedings{fan2020fsod,
    title={Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector},
    author={Fan, Qi and Zhuo, Wei and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={CVPR},
    year={2020}
  }
  ```




