
This is the code repo of SLN-Amodal. Paper could be downloaded [here](https://arxiv.org/abs/1905.12898).

This repository is in large parts based on Multimodallearning's [Mask_RCNN](https://github.com/multimodallearning/pytorch-mask-rcnn),  we also borrow the amodal evaluation code from [AmodalMask](https://github.com/Wakeupbuddy/amodalAPI) and [COCO API](https://github.com/cocodataset/cocoapi).  The training and evaluation dataset are referenced from [COCOA](https://arxiv.org/abs/1509.01329) and [D2SA](https://arxiv.org/abs/1804.08864).  We would like to thank each of them for their kindly work.

# SLN

In this work, we demonstrate yet another approach to tackle the amodal segmentation problem. Specifically, we first introduce a new representation, namely a semantics-aware distance map (sem-dist map), to serve as our target for amodal segmentation instead of the commonly used masks and heatmaps. The sem-dist map is a kind of level-set representation, of which the different regions of an object are placed into different levels on the map according to their visibility. It is a natural extension of masks and heatmaps, where modal, amodal segmentation, as well as depth order information, are all well-described. Then we also introduce a novel convolutional neural network (CNN) architecture, which we refer to as semantic layering network, to estimate sem-dist maps layer by layer, from the global-level to the instance-level, for all objects in an image. Extensive experiments on the COCOA and D2SA datasets have demonstrated that our framework can predict amodal segmentation, occlusion and depth order with state-of-the-art performance.

![](https://github.com/apchenstu/SLN-Amodal/blob/master/results/sem-dist-map-demo.png)

## Authors: 
[Ziheng Zhang*](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Anpei Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Ling Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+L), [Jingyi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Shenghua Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+S)
## Set up environment
 1. We specify pytorch==0.4.0

    Or you can use Ananconda to create new environment(strongly recommanded) in root directory by
    ```bash
    conda create -n SLN-env
    conda install pytorch=0.4.0 cuda90 -c pytorch
    conda install -c conda-forge scikit-image
    conda install -c anaconda cudatoolkit==9.0
    conda install tqdm
    pip install tensorboardX
    ```
    
 2. Configure [COCOAPI](https://github.com/cocodataset/cocoapi)

    We modify the [COCOAPI](https://github.com/cocodataset/cocoapi) to meet our need. Here you have to soft link the pycocotool to the root directory for invoking.
    ```bash
    ln -s /path/to/pycocotool /path/to/our/root/diectory
    ```

## Datasets
* Download pre-trained  [weights](https://drive.google.com/open?id=1ZCeAXqRbsoJDdaJNMWu1uCEfxtk8g9rB) and unzip the package to the root directory.
* we provide layer based [D2SA](https://drive.google.com/open?id=1Y3fHrEmtfri3vZt3ehahL76EXtkfijtN) and [COCOA](https://drive.google.com/open?id=1jLT4zODCoXfO7U6bc-w171xty8fGCY8t) dataset.
We also provide some [scripts](https://github.com/apchenstu/SLN-Amodal/tree/master/scripts) that you can convert the original amodal annotation [[COCOA](https://drive.google.com/file/d/0B8e3LNo7STslUGRFUVlQSnZRUVE/view?usp=drive_open),[D2SA](https://www.mvtec.com/company/research/datasets/mvtec-d2s/)] to our layer based annotation. If you find those two dataset are useful, please cite their [COCOA](https://arxiv.org/abs/1509.01329) and [D2SA](https://arxiv.org/abs/1804.08864) papers.
* [BaiduYunPan](https://pan.baidu.com/s/1zyJAXZmNw5lFdr3DBuaYiQ) link with verify code:yr2i
* Folder structure
  ```
	├──  datasets                       - dataset folder
	│    └── coco_amodal 
	|        └── annotations    
	|            └── COCO_amodal_val[train]2014.json
	|        └── val2014             
	|            └── ###.jpg ###.npz 
    |        └── train2014           
	|            └── ###.jpg ###.npz 
	│    └── D2S                
	|        └── annotations    
	|            └── COCO_amodal_val[train]2014.json
	|        └── val2014             
	|            └── ###.jpg ###.npz 
    |        └── train2014           
	|            └── ###.jpg ###.npz 
	├──  checkpoints                 
    |    └──COCOA[D2SA,deeplabv2,mask_rcnn_coco].pth     
	├──  pycocotool                     - soft link to cocoapi/PythonAPI/pycocotool              
  ```



## Usage

* For training,

  ```bash
  python amodal_train.py train --dataset ./datasets/coco_amodal --model coco
  python amodal_train.py train --dataset ./datasets/D2S --model coco
  ```

* For evaluate,
  ```bash
  python amodal_train.py evaluate --dataset ./datasets/coco_amodal --model ./checkpoints/COCOA.pth --data_type COCOA
  python amodal_train.py evaluate --dataset ./datasets/D2S --model ./checkpoints/D2SA.pth --data_type D2SA
  ```
<!-- 
  * Note: if you want to evaluate the pre-train models,
For COCOA dataset, please make sure [L10](https://github.com/apchenstu/SLN-Amodal/blob/master/amodal_train.py#L10) is
    
    ```bash
    from evaluate.amodalevalCOCOA import AmodalEval
    ```
    For D2SA dataset, please make sure [L10](https://github.com/apchenstu/SLN-Amodal/blob/master/amodal_train.py#L10) is
    ```bash
    from evaluate.amodalevalD2SA import AmodalEval
    ```
-->
  
 * For test images,
    ```bash
    python amodal_test.py 
    ```
    you can modify the path to your image folder inside the script.

## Citation

If you find this code useful to your research, please consider citing:
```tex
@inproceedings{zhang2019amodal,
  title={Learning Semantics-aware Distance Map with Semantics Layering Network for Amodal Instance Segmentation},
  author={Zhang, Zi-Heng and Chen, An-Pei and Xie, Ling and Yu, Jing-Yi and Gao, Sheng-Hua},
  booktitle={2019 ACM Multimedia Conference on Multimedia Conference},
  pages={1--9},
  year={2019},
  organization={ACM}
}
```
