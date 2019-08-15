This is the code repo of SLN-Amodal. Paper could be downloaded [here](https://arxiv.org/abs/1905.12898).

This repo has not been finished up. 

## Author: 

[Ziheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Anpei Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Ling Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+L), [Jingyi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J), [Shenghua Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+S)

### Set up environment

We specify pytorch==0.4.0

Or you can use Ananconda to create new environment in root directory by 

```bash
conda create -n SLN-env --file SLNenv.txt
```


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
python exp_train.py  # for training.
```

```bash
python exp_test.py  # for testing
```

### Citation

Please cite our paper for any purpose of usage.


