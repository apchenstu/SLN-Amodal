"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""


import os, re
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


from tensorboardX import SummaryWriter
from modal.loss import *
from modal.modals import ResNet, FPN, RPN, Classifier, Mask

from modal.Functions import *
from modal.Functions import compose_image_meta, mold_image, log, load_image_gt, build_rpn_targets

from modal.loss import compute_rpn_class_loss, compute_rpn_bbox_loss, compute_mrcnn_bbox_loss, compute_mrcnn_mask_loss, \
    compute_mrcnn_class_loss
from modal.modals import pyramid_roi_align_image



class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, augment=True):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
            """
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[image_index]
        image, image_metas, gt_class_ids, gt_boxes, gt_layer = \
            load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                          use_mini_mask=self.config.USE_MINI_MASK)

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            return None

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_layer = gt_layer[:, :, ids]

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        images = mold_image(image.astype(np.float32), self.config)

        # Convert
        image_raw = torch.from_numpy(image.copy().transpose(2, 0, 1) / 255)
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()

        gt_layer = torch.from_numpy(gt_layer.transpose(3, 2, 0, 1))
        # gloable_layer = torch.from_numpy(gloable_layer.transpose(2, 0, 1))
        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_layer, image_raw

    def __len__(self):
        return self.image_ids.shape[0]


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weightsbbox
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.writer = []
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []
        self.current_epoch = []
        self.layer_decoder = None
        self.amodal_refine = None

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

        # Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(),
                                requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        if self.config.DATA_TYPE == 'amodal':
            self.mask_vis = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

        os.makedirs(config.EXPERIMENT_DIR, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.EXPERIMENT_DIR, purge_step=0)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}".format(self.config.NAME.lower()))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

        # self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn.pth")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "new": r"(fpn.C1.*)|(classifier.*)|(mask.*)|(layer_decoder.*)|(rpn.*)",
            "rpn": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)",
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(layer_decoder.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(layer_decoder.*)|(amodal_refine.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(layer_decoder.*)|(amodal_refine.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(layer_decoder.*)|(amodal_refine.*)",
            "layer": r"(mask.*)|(layer_decoder.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True,
                                                      num_workers=4)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch + 1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(epochs):
            log("Epoch {}/{}.".format(epoch, epochs))
            self.current_epoch += 1

            # Training
            self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)
            torch.save(self.state_dict(), self.checkpoint_path.format(self.epoch))

            self.epoch += 1

    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        step = 0

        optimizer.zero_grad()
        # self.layer_depth_optimizers.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_layer = inputs[6]  # mask, vis_mask, invis_mask
            image_raw = inputs[7]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(images, requires_grad=False)
            rpn_match = Variable(rpn_match, requires_grad=False)
            rpn_bbox = Variable(rpn_bbox, requires_grad=False)
            gt_class_ids = Variable(gt_class_ids, requires_grad=False)
            gt_boxes = Variable(gt_boxes, requires_grad=False)
            gt_layer = Variable(gt_layer, requires_grad=False)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()

            # Run object detection
            data = self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_layer], mode='training')
            if len(data) == 0:
                print('training purpose bbox equal 0.')
                continue
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, unet_features, image_path, amodal_mask, gloable_lab, final_out = data
            isStuff = target_class_ids  # target_class_ids[target_class_ids>0] = 1;target_class_ids[target_class_ids==2] = 1

            # Compute losses
            layer_loss, vis_pred, y_true = compute_layer_loss(target_mask, target_class_ids, mrcnn_mask)

            if len(vis_pred):
                mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
                rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
                rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
                mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
                amodal_loss, amodal_pred = compute_amodal_loss(target_mask, target_class_ids, mrcnn_mask)

                positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
            else:
                continue

            loss = layer_loss + rpn_bbox_loss + mrcnn_bbox_loss + mrcnn_class_loss + amodal_loss + rpn_class_loss

            # Backpropagation
            # if layer_loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                # if self.layer_netG is not None:
                #     self.optimizer_G.step()
                #     self.optimizer_G.zero_grad()
                batch_count = 0

            # Progress
            printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                             suffix="Complete - loss: {:.5f} ".format(loss.data.cpu().item()), length=100)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def detect(self, images, mode='inference'):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        with torch.no_grad():
            molded_images, image_metas, windows = self.mold_inputs(images)

            # Convert images to torch tensor
            molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

            # To GPU
            if self.config.GPU_COUNT:
                molded_images = molded_images.cuda()

            # Wrap in variable
            molded_images = Variable(molded_images)

            # Run object detection
            detections, mrcnn_mask = self.predict([molded_images, image_metas], mode=mode)

            if len(detections) == 0:
                return []

            # Convert to numpy
            detections = detections.data.cpu().numpy()
            mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

            # t = time.time()
            # Process detections
            results = []
            for i, image in enumerate(images):
                final_rois, final_class_ids, final_scores, final_masks = \
                    self.unmold_detections(detections[i], mrcnn_mask[i],
                                           image.shape, windows[i])
                results.append({
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                })
            # print(time.time()-t)
        return results

    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)


        _, _, H, W = molded_images.size()
        GLM_input = F.upsample(molded_images, size=(513, 513), mode="bilinear", align_corners=False)

        self.GLM_modual.eval()
        logits = self.GLM_modual(GLM_input)
        probs = F.softmax(logits, dim=1)
        gloable_lab_small = torch.argmax(probs, dim=1)
        probs = torch.cat((probs, gloable_lab_small.unsqueeze(0).float() / 255), dim=1)
        gloable_lab = F.upsample(gloable_lab_small.unsqueeze(0).float(), size=(H, W), mode="bilinear",
                                 align_corners=False)

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections, keep = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            if len(detections) == 0:
                return [], []

            detections[:, :2] -= 0
            detections[:, 2:4] += 0
            detections[detections < 0] = 0.0
            detections[detections > 1024] = 1024

            cls_feature = pyramid_roi_align_image([detections[:, :4].view(-1, 4).unsqueeze(0)] + [probs], 16, (65, 65),
                                                  istrain=False)
            cls_feature = Variable(cls_feature)

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask, mrcnn_out_feature = self.mask(mrcnn_feature_maps, detection_boxes, cls_feature)

            if self.layer_decoder is not None:

                mrcnn_mask, _ = self.layer_decoder(mrcnn_feature_maps, detection_boxes, cls_feature)
                mrcnn_mask = mrcnn_mask.squeeze(1)

            else:
                mrcnn_mask[:, 1] = F.sigmoid(mrcnn_mask[:, 1:].sum(dim=1))
                # mrcnn_mask = F.sigmoid(mrcnn_mask)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, mrcnn_mask]


        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if len(rois) == 0:
                return []

            image_path = pyramid_roi_align_image([rois.view(-1, 4)] + [molded_images], 32, self.config.IMAGE_SHAPE,
                                                 istrain=True) / 140.0

            if len(rois) > self.config.MAX_NUMB_RPNS:
                ids = np.random.choice(
                    np.arange(len(rois)), self.config.MAX_NUMB_RPNS, replace=False)
                rois = rois[ids]
                target_mask = target_mask[ids]
                target_deltas = target_deltas[ids]
                target_class_ids = target_class_ids[:, :, ids]
                image_path = image_path[ids]

            image_path = image_path.detach()

            GLM_feature = pyramid_roi_align_image([rois.view(-1, 4)] + [probs], 16, (65, 65), istrain=True)
            GLM_feature = Variable(GLM_feature)

            if not rois.size(0):
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_mask_vis = mrcnn_mask
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections

                mrcnn_mask, mrcnn_out_feature = self.mask(mrcnn_feature_maps, rois, GLM_feature)

            unet_features = (mrcnn_feature_maps, rois, GLM_feature)

            if self.layer_decoder is not None and target_class_ids.size(0):
                amodal_mask = F.sigmoid(
                    refinement3(self.layer_decoder, unet_features, mrcnn_mask, target_class_ids, target_mask))
            else:
                amodal_mask = []


            out = []
            if self.amodal_refine:
                gloable_lab_small_piror = pyramid_roi_align_image(
                    [rois.view(-1, 4)] + [gloable_lab_small.unsqueeze(0).float() / 255], 32, (65, 65), istrain=True)
                finetune_feature = torch.cat((image_path, mrcnn_mask, mrcnn_mask, gloable_lab_small_piror.detach()),
                                             dim=1)
                out = self.amodal_refine(finetune_feature)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, unet_features, image_path, amodal_mask, gloable_lab,
                    out]

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        class_ids[class_ids > 0] = 1  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        scales = np.array([h_scale, w_scale, h_scale, w_scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

