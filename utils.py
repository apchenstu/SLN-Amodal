"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.color
import skimage.io
import torch
import pickle
from itertools import chain
from skimage import morphology
from skimage.measure import regionprops
############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        box = np.array([y1, x1, y2, x2])+(np.random.rand(4)*2-1)*(y2-y1,x2-x1,y2-y1,x2-x1)/15
        box[box<0] = 0
        boxes[i] =  box
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.name_to_id = {}
        for i in range(len(self.class_names)):
            self.name_to_id[self.class_names[i]] = i

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_name_class_id(self, class_name):
        """Takes a  class name and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.beach") -> 23
        """

        return self.name_to_id['foreground']
        #return self.name_to_id[class_name]

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # print( self.image_info[image_id]['path'])
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_features(self,image_id):
        """Load the pretrain features and return a [N,H,W] Numpy array.
        """
        # Load feature
        image_path = self.image_info[image_id]['path']
        with open(image_path[:-4]+'.out', 'rb') as fp:
            Features = pickle.load(fp)

        return Features

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    # h, w = image.shape[:2]
    # window = (0, 0, h, w)
    # scale = 1
    #
    # # Scale?
    # if min_dim:
    #     # Scale up but not down
    #     scale = max(1, min_dim / min(h, w))
    # # Does it exceed max dim?
    # if max_dim:
    #     image_max = max(h, w)
    #     if round(image_max * scale) > max_dim:
    #         scale = max_dim / image_max
    # # Resize image and mask
    # if scale != 1:
    #     image = scipy.misc.imresize(
    #         image, (round(h * scale), round(w * scale)))
    # # Need padding?
    # if padding:
    #     # Get new height and width
    #     h, w = image.shape[:2]
    #     top_pad = (max_dim - h) // 2
    #     bottom_pad = max_dim - h - top_pad
    #     left_pad = (max_dim - w) // 2
    #     right_pad = max_dim - w - left_pad
    #     padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    #     image = np.pad(image, padding, mode='constant', constant_values=0)
    #     window = (top_pad, left_pad, h + top_pad, w + left_pad)
    # return image, window, scale, padding

    h, w = image.shape[:2]
    image = scipy.misc.imresize( image, (max_dim, max_dim))
    window = (0,0,max_dim,max_dim)
    scale = (max_dim/h,max_dim/w)
    padding = [(0, 0), (0, 0), (0, 0)]
    return image, window, scale, padding

def resize_layer(mask,scale,padding):
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale[0], scale[1], 1,1], order=0)
    #mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale[0], scale[1], 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[..., i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        mini_mask[..., i] = np.where(m > 0, 1, 0)
    return mini_mask




def minimize_labal(bbox, labal, mini_shape,channle=8):
    """id start from 1
    """
    labal_ids = get_image_labals(labal)
    mini_mask = np.zeros(mini_shape + (bbox.shape[0],channle), dtype='uint8')
    for depth in range(channle):
        (mask,objectID) = layer_to_mask(labal, depth, labal_ids)

        if len(objectID) == 0:
            continue

        big_labal = np.zeros(labal.shape, dtype='uint8')
        for j in range(len(objectID)):
            if not objectID[j] is None:
                big_labal[mask[j]] = objectID[j]+1

        for i in range(bbox.shape[0]):
            y1, x1, y2, x2 = bbox[i][:4]
            m = big_labal[y1:y2, x1:x2,0]
            if m.size == 0:
                raise Exception("Invalid bounding box with area of zero")
            m = scipy.misc.imresize(m, mini_shape, interp='nearest')
            mini_mask[:, :, i,depth] = m
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    mask = mask.squeeze()
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32)/ 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def reLayerMask(mask_amodal, mask_invis):
    mask_zeros = np.zeros(mask_amodal[0].shape).astype('bool')
    labal = np.zeros(mask_amodal[0].shape).astype('uint64')
    for i in range(len(mask_amodal)):
        if i >= 32:
            continue
        if len(mask_invis[i]):
            invis = mask_invis[i] > 0
            labal[invis] |= 1 << (i + 32)
            mask_vis = mask_amodal[i] - mask_invis[i]
        else:
            mask_vis = mask_amodal[i]

        labal[mask_vis > 0] |= 1 << i

    labal = remove_small_path(labal, min_size=64)
    return labal


def remove_small_path(labal, min_size=64):
    color = np.unique(labal)
    for i in range(len(color)):
        mask = (labal == color[i])
        mask_new = morphology.remove_small_objects(mask, min_size=min_size)
        if not mask_new.max():
            labal[mask] = 0
    return labal


def get_image_labals(labal):
    labal_ids = np.unique(labal)
    if labal_ids[0] == 0:
        labal_ids = np.delete(labal_ids, 0)
    return labal_ids


# id start from 0
def objectID_to_masks(labal, id, labal_ids=None):
    if labal_ids is None:
        labal_ids = get_image_labals(labal)

    mask_vis, mask_invis = [], []
    index_vis = ((labal_ids >> id) & 1 == 1).nonzero()
    index_invis = ((labal_ids >> (id + 32)) & 1 == 1).nonzero()

    for items in index_vis:
        if items.size > 0:
            mask_vis.append(labal == labal_ids[items[0]])
    for items in index_invis:
        if items.size > 0:
            mask_invis.append(labal == labal_ids[items[0]])

    return (mask_vis, index_vis, mask_invis, index_invis)


# id start from 0, id<0 return all masks
def maskID_to_mask(labal, id, labal_ids=None):
    if labal_ids is None:
        labal_ids = get_image_labals(labal)

    mask = []
    if id < 0:
        for items in labal_ids:
            mask.append(labal == items)
        return mask
    else:
        return labal == labal_ids[id]


def number_to_index(labal_id):
    bin_index, objectID = 0, []
    while labal_id:
        if labal_id & np.uint64(1):
            return bin_index
        bin_index += 1
        labal_id = labal_id >> np.uint64(1)



def remove_last_one(number, depth):
    while number and depth:
        number = np.bitwise_and(number,np.uint64(number - 1))
        depth -= 1
    return number


# id start from 0
# return vis object id invis layer 1 - n
def maskID_to_objectIDs(labal, id, labal_ids=None):
    if labal_ids is None:
        labal_ids = get_image_labals(labal)
    labal_id = labal_ids[id]

    vis = (labal_id << np.uint64(32)) >> np.uint64(32)  # remove highest 32 bit
    invis = labal_id >> np.uint64(32)  ## remove lowest 32 bit

    object_id_vis = number_to_index(vis)
    object_id_invis = number_to_index(invis)
    object_id_vis.extend(object_id_invis)
    return object_id_vis


def layer_to_mask(labal, depth, labal_ids=None):
    if labal_ids is None:
        labal_ids = get_image_labals(labal)
    mask, objectID = [], []
    if 0 == depth:
        vis = (labal_ids << np.uint64(32)) >> np.uint64(32)
        for i in range(len(vis)):
            mask.append(maskID_to_mask(labal, i))
            objectID.append(number_to_index(vis[i]))
        return (mask, objectID)

    else:
        # find (depth)th 1 from left to right, is exist have depth layer else not
        depth -= 1
        invis = labal_ids >> np.uint64(32)
        for i in range(len(invis)):
            new_labal = remove_last_one(invis[i], depth)
            if new_labal:
                mask.append(maskID_to_mask(labal, i))
                objectID.append(number_to_index(invis[i]))
        return (mask, objectID)



