import os
import skimage.io

import coco
import pickle
import model as modellib
import visualize

import torch
import torch.nn as nn
from modal.deeplabv2 import *


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
AMODAL_MODEL_PATH = './checkpoints/COCOA.pth'


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

IMAGE_DIR = './datasets/coco_amodal/val2014/'

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
num_classes = 1+1

model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
model.mask.conv1 = nn.Conv2d(439, 256, kernel_size=3, stride=1)
model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)
model.GLM_modual = DeepLabV2_ResNet101_MSC(182)
model.current_epoch = 0

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(AMODAL_MODEL_PATH))

if config.GPU_COUNT:
    model = model.cuda()

class_names = ['BG', 'objects']
image_list = os.listdir(IMAGE_DIR)
for i,item in enumerate(image_list):
    if item.endswith('.jpg'):
        print('image: ',item)
        image_path = os.path.join(IMAGE_DIR,item)
        image = skimage.io.imread(image_path)

        results = model.detect([image])

        if len(results)==0:
            continue

        r = results[0]

        save_path = os.path.join('./results/', item+'.json')
        with open(save_path, 'wb') as output:
            pickle.dump(r, output)


