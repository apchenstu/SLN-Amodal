import numpy as np
import random
import cv2,os,pickle
from skimage import morphology

class amodalImage:
    def __init__(self, image_file=None):
        with open(image_file, 'rb') as fp:
            self.Image_anns = pickle.load(fp)

    # use number to label areas, 2**i = visual areas index for amodal i ,2**(32+i) = invisual mask index for amodal i

    def reLayerMask(self,mask_amodal, mask_invis):
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

        labal = self.remove_small_path(labal, min_size=64)
        return labal

    def remove_small_path(self, labal, min_size=64):
        color = np.unique(labal)
        for i in range(len(color)):
            mask = (labal == color[i])
            mask_new = morphology.remove_small_objects(mask, min_size=min_size)
            if not mask_new.max():
                labal[mask] = 0
        return labal

    def get_image_labals(self,labal):
        labal_ids = np.unique(labal)
        if labal_ids[0] == 0:
            labal_ids = np.delete(labal_ids, 0)
        return labal_ids

    # id start from 0
    def objectID_to_masks(self,labal, id, labal_ids=None):
        if labal_ids is None:
            labal_ids = self.get_image_labals(labal)

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
    def maskID_to_mask(self,labal, id, labal_ids=None):
        if labal_ids is None:
            labal_ids = self.get_image_labals(labal)

        mask = []
        if id < 0:
            for items in labal_ids:
                mask.append(labal == items)
        else:
            mask.append(labal == labal_ids[id])
        return mask

    def number_to_index(self,labal_id):
        bin_index, objectID = 0, []
        while labal_id:
            if labal_id & np.uint64(1):
                objectID.append(bin_index)
            bin_index += 1
            labal_id = labal_id >> np.uint64(1)
        return objectID

    def remove_last_one(self,number, depth):
        while number and depth:
            number = number & (number - 1)
            depth -= 1
        return number

    # id start from 0
    # return vis object id invis layer 1 - n
    def maskID_to_objectIDs(self,labal, id, labal_ids=None):
        if labal_ids is None:
            labal_ids = self.get_image_labals(labal)
        labal_id = labal_ids[id]

        vis = (labal_id << np.uint64(32)) >> np.uint64(32)  # remove highest 32 bit
        invis = labal_id >> np.uint64(32)  ## remove lowest 32 bit

        object_id_vis = self.number_to_index(vis)
        object_id_invis = self.number_to_index(invis)
        object_id_vis.extend(object_id_invis)
        return object_id_vis

    def layer_to_mask(self,labal, depth, labal_ids=None):
        if labal_ids is None:
            labal_ids = self.get_image_labals(labal)
        mask, objectID = [], []
        if 0 == depth:
            vis = (labal_ids << np.uint64(32)) >> np.uint64(32)
            for i in range(len(vis)):
                mask.append(self.maskID_to_mask(labal, i))
                objectID.append(self.number_to_index(vis[i]))
            return (mask, objectID)

        else:
            # find (depth)th 1 from left to right, is exist have depth layer else not
            depth -= 1
            invis = labal_ids >> np.uint64(32)
            for i in range(len(invis)):
                new_labal = self.remove_last_one(invis[i], depth)
                if new_labal:
                    mask.append(self.maskID_to_mask(labal, i))
                    objectID.append(self.number_to_index(invis[i]))
            return (mask, objectID)