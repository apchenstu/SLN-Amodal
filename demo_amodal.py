import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import pickle
import model as modellib
import visualize

import torch
import torch.nn as nn
from modal.modals import *
from modal.densenet import DenseNet
from modal.deeplabv2 import *


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
AMODAL_MODEL_PATH = './checkpoints/mask_rcnn_coco_0005.pth'


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
# model.mask.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
# # model.mask_vis.conv5 = New(num_classes=2)
# model.classifier.linear_class = nn.Linear(1024, num_classes)
# model.classifier.linear_bbox = nn.Linear(1024, num_classes * 4)
#
# gpu_ids = [0];lr=2e-4
# model.layer_netG = networks.define_G(num_classes, num_classes, 64,'unet_32', gpu_ids)

model.mask.conv1 = nn.Conv2d(439, 256, kernel_size=3, stride=1)
model.mask.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
model.classifier.linear_class = nn.Linear(1024, num_classes)
model.classifier.linear_bbox = nn.Linear(1024, num_classes * 4)
model.current_epoch = 0

# gpu_ids = [0];lr=2e-4
# model.layer_netG = networks.define_G(config.NUM_CLASSES, config.NUM_CLASSES, 64,'unet_32', gpu_ids)#opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids,False
# model.layer_netD = networks.define_D(3, 64,'basic', 3, gpu_ids)
# model.optimizer_G = torch.optim.Adam(model.layer_netG.parameters(),lr=lr, betas=(0.5, 0.999))
# model.optimizer_D = torch.optim.Adam(model.layer_netD.parameters(),lr=lr, betas=(0.5, 0.999))

model.layer_netG = Mask(439, config.MASK_POOL_SIZE, config.IMAGE_SHAPE,
                        num_classes)  # copy.deepcopy(model.mask)#
model.layer_netG.load_state_dict(model.mask.state_dict())
model.segmentation_module = DeepLabV2_ResNet101_MSC(182)
model.segmentation_module.load_state_dict(torch.load('./checkpoints/deeplabv2.pth'))

# model.denstNet = DenseNet(depth=190, num_classes=100, growthRate=40)
# denstNet_checkpoint = torch.load('./scrips/model_best.pth')
#model.denstNet.load_state_dict(denstNet_checkpoint)

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(AMODAL_MODEL_PATH))

if config.GPU_COUNT:
    model = model.cuda()



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'objects']
#
# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#
# # Run detection
# results = model.detect([image])
#
#
# # Visualize results
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# plt.show()
final_list = ['1533','3525','4154','8480','9514','9463','6341']
class_names = ['BG', 'objects']
image_list = os.listdir(IMAGE_DIR)
for i,item in enumerate(image_list):
    if item.endswith('.jpg'):
        print('image: ',item)
        image_path = os.path.join(IMAGE_DIR,item)
        image = skimage.io.imread(image_path)

        # is_in_list = False
        # for k in final_list:
        #     is_in_list += item.endswith(k+'.jpg')
        # if not is_in_list:
        #     continue

        results = model.detect([image])

        if len(results)==0:
            continue

        r = results[0]

        save_path = os.path.join('./results/', item+'.json')
        with open(save_path, 'wb') as output:
            pickle.dump(r, output)

        # save_path = os.path.join('./results/', item )
        # result = visualize.get_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
        #
        # skimage.io.imsave(save_path,result)

