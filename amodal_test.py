import os
import skimage.io


import pickle
import model as modellib
from modal.deeplabv2 import *
from amodal_train import Amodalfig

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = './datasets/coco_amodal/val2014/'
AMODAL_MODEL_PATH = './checkpoints/COCOA.pth'


class InferenceConfig(Amodalfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

config = InferenceConfig()
config.display()

# Create model object.
config.NUM_CLASSES = 1+1
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
model.mask.conv1 = nn.Conv2d(439, 256, kernel_size=3, stride=1)
model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)
model.GLM_modual = DeepLabV2_ResNet101_MSC(182)
model.current_epoch = 0

# Load weights trained on MS-COCO
model.load_weights(AMODAL_MODEL_PATH)

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


