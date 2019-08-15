This is the code repo of SLN-Amodal.

This repo has not finished up. 

## Author: 

To be done

### Environment Requirement

Follow the following instructions for step by step installation.

#### Requirements

pytorch==0.4.0


#### Install [COCOAPI](https://github.com/cocodataset/cocoapi)
```bash
pip install pycocotool
```
<!-- We use [COCOAPI](https://github.com/cocodataset/cocoapi) to fetch data when experimenting with COCO. The installation for python api could be briefly stated as :

- Go to [COCOAPI](https://github.com/cocodataset/cocoapi) to clone the repo. 

- Type

  ```bash
  make
  ```

  in [pythonAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI) to compile.

- Link the api to root file directory

```bash
ln -s ./cocoapi/PythonAPI/pycocotools ./pycocotools
```
 -->


### Usage

In repo's root directory, type

```bash
python exp_train.py 
```

for training.

```bash
python exp_test.py
```

for testing



### Citation

Please cite our paper for any purpose of usage.


