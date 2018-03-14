import os.path

import cv2
import numpy as np
from PIL import Image

import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation


def eval_one_result(loader, folder, one_mask_per_image=False, mask_thres=0.5, use_void_pixels=True, custom_box=False):
    def mAPr(per_cat, thresholds):
        n_cat = len(per_cat)
        all_apr = np.zeros(len(thresholds))
        for ii, th in enumerate(thresholds):
            per_cat_recall = np.zeros(n_cat)
            for jj, categ in enumerate(per_cat.keys()):
                per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ])

            all_apr[ii] = per_cat_recall.mean()

        return all_apr.mean()

    # Allocate
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["all_percent"] = np.zeros(len(loader))
    eval_result["meta"] = []
    eval_result["per_categ_jaccard"] = dict()

    # Iterate
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        if not one_mask_per_image:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
        else:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.
        gt = np.squeeze(helpers.tens2image(sample["gt"]))
        if use_void_pixels:
            void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Threshold
        mask = (mask > mask_thres)
        if use_void_pixels:
            void_pixels = (void_pixels > 0.5)

        # Evaluate
        if use_void_pixels:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask, void_pixels)
        else:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask)

        if custom_box:
            box = np.squeeze(helpers.tens2image(sample["box"]))
            bb = helpers.get_bbox(box)
        else:
            bb = helpers.get_bbox(gt)

        mask_crop = helpers.crop_from_bbox(mask, bb)
        if use_void_pixels:
            non_void_pixels_crop = helpers.crop_from_bbox(np.logical_not(void_pixels), bb)
        gt_crop = helpers.crop_from_bbox(gt, bb)
        if use_void_pixels:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop) & non_void_pixels_crop)/np.sum(non_void_pixels_crop)
        else:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop))/mask_crop.size
        # Store in per category
        if "category" in sample["meta"]:
            cat = sample["meta"]["category"][0]
        else:
            cat = 1
        if cat not in eval_result["per_categ_jaccard"]:
            eval_result["per_categ_jaccard"][cat] = []
        eval_result["per_categ_jaccard"][cat].append(eval_result["all_jaccards"][i])

        # Store meta
        eval_result["meta"].append(sample["meta"])

    # Compute some stats
    eval_result["mAPr0.5"] = mAPr(eval_result["per_categ_jaccard"], [0.5])
    eval_result["mAPr0.7"] = mAPr(eval_result["per_categ_jaccard"], [0.7])
    eval_result["mAPr-vol"] = mAPr(eval_result["per_categ_jaccard"], np.linspace(0.1, 0.9, 9))

    return eval_result



