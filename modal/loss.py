import torch
import torch.nn.functional as F
from torch.autograd import Variable


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size(0):
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size(0):
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def compute_amodal_loss( target_masks, target_class_ids,pred_masks):

    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]

    # Gather the masks (predicted and true) that contribute to loss
    y_true = target_masks[positive_ix].sum(dim=1)#- target_masks[indices[:,0].data,1]
    y_pred = F.sigmoid(pred_masks[positive_ix,1:].sum(dim=1))


    loss = F.binary_cross_entropy(y_pred, y_true)

    return loss,y_pred




def compute_layer_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if  torch.nonzero(target_class_ids > 0).size(0):

        ##### binary_cross_entropy
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]


        y_true = target_masks[positive_ix]  # amodel,vismask,invismask...
        y_pred = F.sigmoid(pred_masks[positive_ix, 1:])

        loss = F.binary_cross_entropy(y_pred, y_true)
        return loss, y_pred,y_true

    else:
        print('Loss equal 0.')
        return 0, []


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size(0):
        #print(target_masks.size(),target_class_ids.size(),pred_masks.size())
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]


        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()
        y_pred = []

    return loss,y_pred


def compute_invis_loss(amodal_pred,vis_pred,target_masks,target_class_ids):
    positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
    positive_class_ids = target_class_ids[positive_ix.data].long()
    indices = torch.stack((positive_ix, positive_class_ids), dim=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = target_masks[indices[:, 0].data,0] - target_masks[indices[:,0].data,1]

    loss = F.smooth_l1_loss(amodal_pred - vis_pred,y_true)*10
    return loss


def compute_layer_depth_loss(layer_depth,ppm_out):
    # shape 1 * 8 * H ×　Ｗ
    ppm, ppmsup = ppm_out
    size_target = ppm.size(-1)

    layer_depth = layer_depth>0
    layer_depth = F.upsample(layer_depth.float(), size=(size_target, size_target), mode='bilinear')


    loss = F.binary_cross_entropy(ppm,layer_depth) + F.binary_cross_entropy(ppm,layer_depth)
    return loss

def refinement_unet(layer_netG, unet_features, vis_mask, target_class_ids,target_mask):#image_path,

        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        vis_mask = vis_mask[indices[:, 0]]
        target_mask = target_mask[indices[:, 0]]


        amodal_mask = layer_netG(vis_mask)
        amodal_loss,amodal_mask = compute_amodal_loss(target_mask, target_class_ids, amodal_mask)

        return amodal_mask,amodal_loss

def refinement2(layer_netG, unet_features, vis_mask, target_class_ids,target_mask):#image_path,
        mrcnn_feature_maps, rois = unet_features

        amodal_mask = layer_netG(mrcnn_feature_maps, rois)
        amodal_loss,amodal_mask = compute_amodal_loss(target_mask, target_class_ids, amodal_mask)
        return amodal_mask,amodal_loss


def refinement3(layer_netG, unet_features, vis_mask, target_class_ids, target_mask):  # image_path,
    mrcnn_feature_maps, rois,cls_features = unet_features
    amodal_mask,_ = layer_netG(mrcnn_feature_maps, rois,cls_features)
    #amodal_loss, amodal_mask = compute_amodal_loss(target_mask, target_class_ids, amodal_mask)

    return amodal_mask#, amodal_loss

def compute_final_loss(final_out, target_mask):

    return F.binary_cross_entropy(F.sigmoid(final_out), target_mask[:,0].unsqueeze(1))

def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss,y_pred = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,y_pred]