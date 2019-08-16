
This is the code repo of SLN-Amodal. Paper could be downloaded [here](https://arxiv.org/abs/1905.12898).

This repository is in large parts based on Multimodallearning's [Mask_RCNN](https://github.com/multimodallearning/pytorch-mask-rcnn),  we also borrow the amodal evaluation code from [AmodalMask](https://github.com/Wakeupbuddy/amodalAPI) and [COCO API](https://github.com/cocodataset/cocoapi).  The training and evaluation dataset are referenced from [COCOA](https://arxiv.org/abs/1509.01329) and [D2SA](https://arxiv.org/abs/1804.08864).  We want to thank each of them for their kindly work.

![](https://github.com/apchenstu/SLN-Amodal/blob/master/results/sem-dist-map-demo.png)

## Authors: 
[Ziheng Zhang*](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Anpei Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Ling Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+L), [Jingyi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Shenghua Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+S)
## Set up environment

 1. We specify pytorch==0.4.0
 
    Or you can use Ananconda to create new environment in root directory by
    ```bash
    conda create -n SLN-env --file SLNenv.txt
    ```
    
 2. Install [COCOAPI](https://github.com/cocodataset/cocoapi)
 
    please follow the COCOAPI repository to install the data loader api, and then create a soft link from pycocotools folder to our root directory.
    ```bash
    ln -s /path/to/pycocotool /path/to/our/root/diectory
    ```

## Datasets
* Download pre-trained  [weights](https://drive.google.com/open?id=1ZCeAXqRbsoJDdaJNMWu1uCEfxtk8g9rB) and unzip the package to the root directory.
* we provide layer based [D2SA](https://drive.google.com/open?id=1Y3fHrEmtfri3vZt3ehahL76EXtkfijtN) and [COCOA](https://drive.google.com/open?id=1jLT4zODCoXfO7U6bc-w171xty8fGCY8t) dataset.
We also provide some [scripts](https://github.com/apchenstu/SLN-Amodal/tree/master/scripts) that you can convert the original amodal annotation [[COCOA](https://drive.google.com/file/d/0B8e3LNo7STslUGRFUVlQSnZRUVE/view?usp=drive_open),[D2SA](https://www.mvtec.com/company/research/datasets/mvtec-d2s/)] to our layer based annotation. If you find those two dataset are useful, please cite their [COCOA](https://arxiv.org/abs/1509.01329) and [D2SA](https://arxiv.org/abs/1804.08864) papers.
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
  python amodal_train.py train --datasetsets/coco_amodal --model coco
  python amodal_train.py train --datasetsets/D2S --model D2S
  ```

* For evaluate,
  ```bash
  python amodal_train.py evaluate --datasetset ./datasetsets/coco_amodal --model ./checkpoints/COCOA.pth
  python amodal_train.py evaluate --datasetset ./datasetsets/D2S --model ./checkpoints/D2SA.pth
  ```
  * Note: if you want to evaluate the pre-train models,
For COCOA dataset, please make sure,  [L11](https://github.com/apchenstu/SLN-Amodal/blob/master/amodal_train.py#L11) and [L21](https://github.com/apchenstu/SLN-Amodal/blob/master/model.py#L21) are 
    ```bash
    from evaluate.amodalevalCOCOA import AmodalEval
    ```
    For D2SA dataset, please make sure,those two lines are
    ```bash
    from evaluate.amodalevalD2SA import AmodalEval
    ```
 * For test images,
    ```bash
    python amodal_test.py 
    ```
    you can modify the path to your images folder inside the script.
  
## Citation

If you find this code useful to your research, please consider citing:
```
@inproceedings{zhang2019amodal,
  title={Learning Semantics-aware Distance Map with Semantics Layering Network for Amodal Instance Segmentation},
  author={Zhang, Zi-Heng and Chen, An-Pei and Xie, Ling and Yu, Jing-Yi and Gao, Sheng-Hua},
  booktitle={2019 ACM Multimedia Conference on Multimedia Conference},
  pages={1--9},
  year={2019},
  organization={ACM}
}
```
