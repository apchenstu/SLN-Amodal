import os
import time

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from modal import networks
from tqdm import tqdm

from config import Config
import itertools
import model as modellib
import pickle

from modal.deeplabv2 import *
from modal.Functions import *

# Root directory of the project
ROOT_DIR = os.getcwd()
AmodalEval = None
# Path to trained weights file
COCO_MODEL_PATH = "./checkpoints/mask_rcnn_coco.pth"
GLM_MODEL_PATH = "./checkpoints/deeplabv2.pth"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################

class Amodalfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class New(nn.Module):
    def __init__(self, num_classes):
        super(New, self).__init__()
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv5(x)

        return x


############################################################
#  Dataset
############################################################

class AmodalDataset(utils.Dataset):
    def load_amodal(self, dataset_dir, subset, data_type='COCO', year='2014', class_ids=None,
                    class_map=None, return_amodal=True):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO("{}/annotations/{}_amodal_{}{}.json".format(dataset_dir, data_type, subset, year))
        # with open("{}/annotations/{}_amodal_{}{}.layer".format(dataset_dir, data_type, subset, year), 'rb') as fp:
        #     self.layer_labal = pickle.load(fp)

        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        image_ids = sorted(list(coco.imgs.keys()))

        anns = {}
        imgToAnns = {}
        imgs = {}
        regions = []
        if 'annotations' in coco.dataset:
            imgToAnns = {ann['image_id']: [] for ann in coco.dataset['annotations']}
            anns = {ann['id']: [] for ann in coco.dataset['annotations']}
            for ann in coco.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann
                for region in ann['regions']:
                    region['image_id'] = ann['image_id']
                    regions.append(region)

        if 'images' in coco.dataset:
            imgs = {im['id']: {} for im in coco.dataset['images']}
            for img in coco.dataset['images']:
                imgs[img['id']] = img

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.regions = regions
        self.dataset = coco

        # two class only: foreground and background
        self.add_class("coco", 1, 'foreground')

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], iscrowd=None)))
        if return_amodal:
            return coco

    def getAmodalAnnIds(self, imgIds=[]):
        """
        Get amodal ann id that satisfy given fiter conditions.
        :param imgIds (int array): get anns for given imgs
        :return: ids (int array) : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(imgIds) == 0:
            anns = self.dataset['annotations']
        else:
            lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
            anns = list(itertools.chain.from_iterable(lists))
        ids = [ann['id'] for ann in anns]

        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def getMask(self, M):
        return maskUtils.decode([M])

    def getAnnMask(self, ann, w, h):
        if type(ann['segmentation']) == list:
            # polygon
            seg = ann['segmentation']

            img = Image.new("L", (w, h))
            draw = ImageDraw.Draw(img)
            draw.polygon(seg, fill=255)
            amodal_mask = np.asarray(img, dtype="bool")
        else:
            amodal_mask = self.getMask(ann['segmentation'])
        amodal_mask = np.squeeze(amodal_mask)

        if 'invisible_mask' in ann:
            invisible_mask = self.getMask(ann['invisible_mask'])
            return amodal_mask.astype('uint8'), invisible_mask.squeeze().astype('uint8')
        else:
            return amodal_mask.astype('uint8'), np.zeros((h, w)).astype('uint8')

    def load_layer(self, image_id):
        image_info = self.image_info[image_id]
        with open(image_info['path'][:-4] + '.layer', 'rb') as fp:
            layer = pickle.load(fp)

        class_ids, occlude_rates = [], []
        instance_masks, invisiable_mask, visiable_mask = [], [], []
        image_labals = get_image_labals(layer)
        object_len = max_objectID(image_labals)
        for i in range(object_len):
            mask_vis, index_vis, mask_invis, index_invis = objectID_to_masks(layer, i, image_labals)

            mask_vis_all = np.zeros((image_info["height"], image_info["width"])).astype('bool')
            for item in mask_vis:
                mask_vis_all += item

            mask_invis_all = np.zeros((image_info["height"], image_info["width"])).astype('bool')
            if len(index_invis) > 0:
                distancesLayers = np.zeros(len(mask_vis))
                for i in range(len(index_invis)):
                    object_id_vis, object_id_invis = maskID_to_objectIDs(layer, i, image_labals)
                    distancesLayers = np.append(distancesLayers, objIDs_to_sindistanceLayer(object_id_invis, i) + 1)
                    mask_invis_all += mask_invis[i]

            visiable_mask.append(mask_vis_all)
            invisiable_mask.append(mask_invis_all)
            instance_masks.append(mask_vis_all + mask_invis_all)
            class_ids.append(1)

        # Pack instance masks into an array
        if len(class_ids):
            mask = np.stack(instance_masks, axis=2)
            mask_invis = np.stack(invisiable_mask, axis=2)
            mask_vis = np.stack(visiable_mask, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, mask_vis, mask_invis
        else:
            # Call super class to return an empty mask
            return super(AmodalDataset, self).load_mask(image_id)

    def load_layer2(self, image_id, config):
        image_info = self.image_info[image_id]
        layer = np.load(image_info['path'][:-4] + '.npz')['layer']

        class_ids = []
        mask_layers = []
        image_labals = get_image_labals(layer)
        object_len = max_objectID(image_labals)
        for i in range(object_len):
            mask_vis, index_vis, mask_invis, index_invis = objectID_to_masks(layer, i, image_labals)
            mask_layer = np.zeros((image_info["height"], image_info["width"], config.NUM_CLASSES - 1)).astype('bool')

            for item in mask_vis:
                mask_layer[..., 0] += item

            if len(index_invis) > 0:
                for j in range(len(index_invis)):
                    object_id_vis, object_id_invis = maskID_to_objectIDs(layer, index_invis[j], image_labals)
                    distancesLayer = objIDs_to_sindistanceLayer(object_id_invis, i) + 1

                    if distancesLayer >= config.NUM_CLASSES - 1 - 1:
                        mask_layer[:, :, -1] += mask_invis[j]
                    else:
                        mask_layer[..., distancesLayer[0]] += mask_invis[j]

            mask_layers.append(mask_layer)
            class_ids.append(1)

        # Pack instance masks into an array
        if len(class_ids):
            mask_layers = np.stack(mask_layers, axis=3)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask_layers, class_ids
        else:
            # Call super class to return an empty mask
            return super(AmodalDataset, self).load_mask(image_id)

    def getAnnMask2(self, ann, w, h):
        if type(ann['segmentation']) == list:
            # polygon
            seg = ann['segmentation']

            img = Image.new("L", (w, h))
            draw = ImageDraw.Draw(img)
            all_mask = np.asarray(img, dtype="uint8")
        else:
            all_mask = self.getMask(ann['segmentation'])
        all_mask = np.squeeze(all_mask)

        if 'visible_mask' in ann:
            visible_mask = self.getMask(ann['visible_mask'])
            visible_mask = np.squeeze(visible_mask)
            return all_mask, visible_mask.astype('uint8')
        else:
            return all_mask, np.zeros((h, w)).astype('uint8')

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(AmodalDataset, self).load_mask(image_id)

        class_ids, occlude_rates = [], []
        vis_all_mask = np.zeros((image_info["height"], image_info["width"]))
        instance_masks, invisiable_mask, visiable_mask = [], [], []
        annotations = self.image_info[image_id]["annotations"][0]
        if type(annotations) == list:
            print("ann cannot be a list! Should be a dict")
            return 0
        for i, ann in enumerate(annotations['regions']):

            # class_id = ann['category_id']
            if 'isStuff' in ann.keys():
                if ann['isStuff']:
                    class_id = 1
                else:
                    class_id = 1
            else:
                class_id = 1
            # class_id = 1

            if 'occlude_rate' in ann.keys():
                occlude_rate = ann['occlude_rate']
                occlude_rates.append(occlude_rate)

            if class_id:
                m, invisible_m = self.getAnnMask(ann, image_info["width"], image_info["height"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

            # vis_all_mask[m-invisible_m] = i
            instance_masks.append(m)
            invisiable_mask.append(invisible_m)
            visiable_mask.append(m - invisible_m)
            class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            mask_invis = np.stack(invisiable_mask, axis=2)
            mask_vis = np.stack(visiable_mask, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, mask_vis, mask_invis
        else:
            # Call super class to return an empty mask
            return super(AmodalDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(AmodalDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):

            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            if class_ids[i] > 0:
                class_id = 1
            else:
                class_id = 0

            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evalute_amodal(amodalGt, model, limit=-1, image_ids=None):
    # Pick COCO images from the dataset
    image_ids = image_ids or amodalGt.image_ids
    if 'COCOA' == args.data_type:
        from evaluate.amodalevalCOCOA import AmodalEval
    else:
        from evaluate.amodalevalD2SA import AmodalEval

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [amodalGt.image_info[id]["id"] for id in image_ids]

    results = []
    pbar = tqdm(total=len(image_ids))
    for i, image_id in enumerate(image_ids):
        # Load image
        image = amodalGt.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]

        # Convert results to COCO format
        image_results = build_coco_results(amodalGt, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)
        pbar.update(1)
    if len(results) == 0:
        return

    # with open('results.json', 'wb') as output:
    #     pickle.dump(results, output)

    # Load results. This modifies results with additional attributes.
    coco_results = amodalGt.dataset.loadRes(results)

    amodalEval = AmodalEval(amodalGt, coco_results, limit)
    print_result(amodalEval)


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    if len(results) == 0:
        return

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    print_result(cocoEval)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=-1,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)

    parser.add_argument('--layer_arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--layer_weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--layer_fc_dim', default=256, type=int,
                        help='number of features between encoder and decoder')
    parser.add_argument('--layer_lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--layer_beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--layer_weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--data_type', default='COCOA', type=str,
                        help='data type, COCOA or D2SA')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = Amodalfig()
    else:
        class InferenceConfig(Amodalfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    model.epoch, model.best = 0, 0
    if args.command == 'train':
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path)

    config.NUM_CLASSES = 1 + 1  # background,layers
    model.mask.conv1 = nn.Conv2d(439, 256, kernel_size=3, stride=1)
    model.mask.conv5 = nn.Conv2d(256, config.NUM_CLASSES, kernel_size=1, stride=1)
    model.classifier.linear_class = nn.Linear(1024, config.NUM_CLASSES)
    model.classifier.linear_bbox = nn.Linear(1024, config.NUM_CLASSES * 4)
    model.current_epoch = 0

    model.GLM_modual = DeepLabV2_ResNet101_MSC(182)
    model.GLM_modual.load_state_dict(torch.load(GLM_MODEL_PATH))


    networks.print_network(model.mask)
    networks.print_network(model.classifier)
    networks.print_network(model.GLM_modual)

    if args.command != 'train':
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path)

    if config.GPU_COUNT:
        model = model.cuda()

    # Train or evaluate
    if args.command != "evaluate":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = AmodalDataset()
        dataset_train.load_amodal(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = AmodalDataset()
        dataset_val.load_amodal(args.dataset, "val", year=args.year)
        dataset_val.prepare()

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=2,
                          layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=3,
                          layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 10,
                          epochs=1,
                          layers='all')



    elif args.command == "evaluate":
        dataset_val = AmodalDataset()
        dataset_val.load_amodal(args.dataset, "val", year=args.year)
        dataset_val.prepare()
        evalute_amodal(dataset_val, model, limit=args.limit,args=args)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
