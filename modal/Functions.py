import torch
import numpy as np
import utils, random

from nms.nms_wrapper import nms
from torch.autograd import Variable
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import matplotlib.cm as cm
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


############################################################
#  Logging Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if iteration == total:
        print()


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :]  # TODO: Support batch size > 1 ff.
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes


############################################################
#  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps




def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    if torch.nonzero(gt_class_ids < 0).size(0):
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[:, crowd_ix.data, :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[:, non_crowd_ix.data]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size(0):
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)

        # Assign positive ROIs to GT masks
        # gt_masks,ids = reMask(gt_masks,boxes)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        roi_masks = gt_masks[:, roi_gt_box_assignment.data, :, :]

        box_ids = Variable(torch.arange(roi_masks.size()[1]), requires_grad=False).int()
        #　roi_masks = lababTOmask(roi_gt_class_ids, roi_masks)
        # roi_gt_class_ids[:] = 1

        if config.GPU_COUNT:
            box_ids = box_ids.cuda()
            roi_masks = roi_masks.cuda()

        mask = [CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks[i].unsqueeze(1).float(),
                                                                                     boxes, box_ids).data for i in
                range(roi_masks.size(0))]
        masks = Variable(torch.stack(mask, dim=1), requires_grad=False)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks.squeeze(2))
    else:
        positive_count = 0
        ids = []

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size() and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, masks.size(1), config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                         requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, masks.size(1), config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                         requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    # masks = re_target_mask(masks,ids)
    #　print(len(roi_gt_class_ids))
    return rois, roi_gt_class_ids, deltas, masks


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes


def coordinate_convert(rois, deltas_specific, config, use_cuda=False):
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if use_cuda:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if use_cuda:
        scale = scale.cuda()
    refined_rois *= scale
    return refined_rois


def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    refined_rois = coordinate_convert(rois, deltas_specific, config, config.GPU_COUNT)

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes

    keep_bool = class_ids > 0
    if config.USE_NMS:
        # Filter out low confidence boxes
        if config.DETECTION_MIN_CONFIDENCE:
            keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

        if max(keep_bool) == 0:
            return [], []
        keep = torch.nonzero(keep_bool)[:, 0]

        # Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            # Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

            # Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data, :]

            class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

            # Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i == 0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)
    else:

        keep = torch.nonzero(keep_bool).view((-1))

        if len(keep) > 100:
            ix_scores, order = class_scores[keep.data].sort(descending=True)
            keep = keep[order[:100]]

        # else:
        #     ix_scores, order = class_scores[~keep_bool].sort(descending=False)
        #     keep2 = torch.nonzero(~keep_bool).view((-1))[order[:(1000-len(keep))]]
        #     keep = torch.cat((keep,keep2),0)

    ix_scores, order = class_scores[keep.data].sort(descending=True)
    keep = keep[order]

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES

    if len(keep.data) > 0:
        top_ids = class_scores[keep.data].sort(descending=True)[1][:]
        keep = keep[top_ids.data]

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are in image domain.
        result = torch.cat((refined_rois[keep.data],
                            class_ids[keep.data].unsqueeze(1).float(),
                            class_scores[keep.data].unsqueeze(1)), dim=1)
    else:
        return [], []

    return result, keep


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections, keep = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections, keep


def detection_inference_bbox(detections, gloable_lab, out_number=20):
    """" input
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """
    # new_detections = torch.Tensor((out_number,detections.size(1)))
    box = []
    gloable_lab = gloable_lab.cpu().numpy()
    props = regionprops(gloable_lab[0])
    for prop in props:
        area = prop.bbox
        a = (area[2] - area[0]) * (area[3] - area[1])
        if a > 16 and a < gloable_lab.shape[1] * gloable_lab.shape[2]:
            box.append(torch.tensor(area + (1, 1)))
    if len(box):
        box = torch.stack(box, dim=0).float()
        box[:, :4] = torch.floor(box[:, :4] / gloable_lab.shape[2] * 1024)
        box = box.cuda()
        return torch.cat((detections, box), dim=0)
    else:
        return detections

    bbox_num = detections.size(0)
    index = np.random.randint(0, bbox_num, out_number)
    new_detections = [detections[index[i]] for i in range(out_number)]
    new_detections = torch.stack(new_detections, dim=0)
    return new_detections
    # for i in range(out_number):
    #     detections[index[i]]


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]  # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask

    image = dataset.load_image(image_id)

    mask_layers, class_ids = dataset.load_layer2(image_id, config)

    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    mask_layers = utils.resize_layer(mask_layers, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask_layers = np.fliplr(mask_layers)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    amodal_mask = np.sum(mask_layers,axis=2)
    bbox = utils.extract_bboxes(amodal_mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([128], dtype=np.int32)
    active_class_ids[range(128)] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask_layers = utils.minimize_mask(bbox, mask_layers, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    mask_layers = (np.swapaxes(mask_layers, 2, 3)>0).astype('uint8')
    return image, image_meta, class_ids, bbox, mask_layers


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def reMask(mask, bbox):
    # mask shape: 2*objects channel*H*W
    # y1, x1, y2, x2
    shape = mask.size()
    lab = torch.zeros((shape[2], shape[3]))
    # for i in range(shape[1]-1,-1,-1):
    #     lab[mask[1,i]>0] = i+1

    for i in range(shape[1]):
        m = mask[0, i] > 0
        if torch.sum(m) < 0.1 * shape[2] * shape[3]:
            continue
        lab[m] = i + 1

    lab = lab.cuda()
    layer_ids = []
    bbox = (torch.tensor([shape[2], shape[3], shape[2], shape[3]]).float().cuda() * bbox).long()
    for i in range(bbox.size(0)):
        ids = np.trim_zeros(np.unique(lab[bbox[i, 0]:bbox[i, 2], bbox[i, 1]:bbox[i, 3]].cpu().numpy()))
        layer_ids.append(torch.from_numpy(ids).cuda())

    mask = mask.float()
    for i in range(shape[1]):
        mask[1, i] = lab
    return mask, layer_ids


def re_target_mask(masks, ids):
    for i in range(len(ids)):
        if len(ids[i]):
            masks[i, 1] = masks[i, 1] == ids[i][-1]
    return masks


def colorize(labelmap):
    # Assign a unique color to each label
    labelmap = labelmap / 152
    colormap = cm.jet_r(labelmap)[..., :-1]
    return colormap.squeeze(0)


def clip_boundary(deeplab_input, gloable_lab):
    mask = (deeplab_input[:, 0] == -123.7) - (deeplab_input[:, 1] == 116.8) - (deeplab_input[:, 2] == 103.9)
    gloable_lab[0, mask] = 255
    return gloable_lab


def print_format(ap, iouThr, stats, areaRng, maxDets):
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'

    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(0.5, 0.95) \
        if iouThr is None else '{:0.2f}'.format(iouThr)
    print(iStr.format(titleStr, typeStr, iouStr, str(areaRng[0]), maxDets, stats))


def print_eval_result(evalute_amodal):
    stats = evalute_amodal.stats
    print_format(1, None, stats[0], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[-1])
    print_format(1, 0.5, stats[1], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[-1])
    print_format(1, 0.75, stats[2], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[-1])

    print_format(0, None, stats[3], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[0])
    print_format(0, None, stats[4], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[1])
    print_format(0, None, stats[5], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[2])
    # print_format(0, None, stats[6], evalute_amodal.params.areaRng, evalute_amodal.params.maxDets[3])


def print_result(amodalEval):
    print('######################### both  #################\n')
    amodalEval.params.onlyThings = 0
    amodalEval.params.occRng = [0, 1000]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 0
    amodalEval.params.occRng = [0, 0.00001]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 0
    amodalEval.params.occRng = [0.00001, 0.25]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 0
    amodalEval.params.occRng = [0.25, 1]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    #####
    print('######################### thing  #################\n')
    amodalEval.params.onlyThings = 1
    amodalEval.params.occRng = [0, 1000]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 1
    amodalEval.params.occRng = [0, 0.00001]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 1
    amodalEval.params.occRng = [0.00001, 0.25]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 1
    amodalEval.params.occRng = [0.25, 1]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    #####
    print('######################### stuff only  #################\n')
    amodalEval.params.onlyThings = 2
    amodalEval.params.occRng = [0, 1000]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 2
    amodalEval.params.occRng = [0, 0.00001]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 2
    amodalEval.params.occRng = [0.00001, 0.25]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)

    amodalEval.params.onlyThings = 2
    amodalEval.params.occRng = [0.25, 1]
    amodalEval.evaluate()
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    print_eval_result(amodalEval)


#######################################   data decoder    ###########################################
# get how many small picese
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
    index_vis = np.where((labal_ids >> id) & 1 == 1)[0]
    index_invis = np.where((labal_ids >> np.uint64(id + 32)) & 1 == 1)[0]

    for items in index_vis:
        mask_vis.append(labal == labal_ids[items])
    for items in index_invis:
        mask_invis.append(labal == labal_ids[items])

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
    if 0 == labal_id:
        return None

    while labal_id:
        if labal_id & np.uint64(1):
            objectID.append(bin_index)
        bin_index += 1
        labal_id = labal_id >> np.uint64(1)
    return objectID


def objIDs_to_sindistanceLayer(object_ids, objID):
    return np.where(np.array(object_ids) == objID)[0]


def remove_last_one(number, depth):
    while number and depth:
        number = np.bitwise_and(number, np.uint64(number - 1))
        depth -= 1
    return number


def max_objectID(labal_ids):
    shift = 0
    vis = (labal_ids << np.uint64(32)) >> np.uint64(32)
    while len(np.where(vis >> np.uint64(shift) == 1)[0]) > 0:
        shift += 1
    return shift


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

    return object_id_vis, object_id_invis


def layer_to_mask(labal, depth, labal_ids=None):
    if labal_ids is None:
        labal_ids = get_image_labals(labal)
    mask, objectID = [], []
    vis = (labal_ids << np.uint64(32)) >> np.uint64(32)
    if 0 == depth:
        for i in range(len(vis)):
            mask.append(maskID_to_mask(labal, i))
            objectID.append(number_to_index(vis[i]))
        return (mask, objectID)

    else:
        # find (depth)th 1 from left to right, is exist have depth layer else not
        depth -= 1
        labal_inds = []
        invis = labal_ids >> np.uint64(32)
        for i in range(len(invis)):  # len(invis)
            new_labal = remove_last_one(invis[i], depth)

            if new_labal:

                object_id = number_to_index(new_labal)  # the first vis object index
                _, _, mask_invis, index_invis = objectID_to_masks(labal, object_id, labal_ids)

                for k in range(len(index_invis)):
                    labal_ind = labal_ids[index_invis[0][k]]

                    new_labal_2 = remove_last_one(labal_ind >> np.uint64(32), depth)
                    if new_labal_2 and object_id == number_to_index(new_labal_2):
                        if labal_ind not in labal_inds:
                            labal_inds.append(labal_ind)
                            mask.append(mask_invis[k])
                            objectID.append(object_id)

        return (mask, objectID)
