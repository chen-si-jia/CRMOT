# 功能：CRTracker 单视图推理 demo
# Function: CRTracker for single view inference demo
# 撰写人：陈思佳，华中科技大学
# 时间：20260125


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import torch.nn as nn

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing, plot_mot_tracking_online_add_conf
from opts import opts


from utils.post_process import ctdet_post_process
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from collections import defaultdict
from tqdm import tqdm
from deep_sort.mvtracker import MVTracker
from deep_sort.update import Update


from PIL import Image
from torchvision import transforms
from tracker.multitracker import JDETracker_to_bbox

from aptm.aptm_module import APTM


def write_results(txt_path, frame_name, bbox_xyxy, track_id):
    save_format = '{frame},{id},{x1},{y1},{x2},{y2}\n'
    with open(txt_path, 'a') as f:
        x1, y1, x2, y2 = bbox_xyxy
        line = save_format.format(frame=frame_name, id=int(track_id), x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
        f.write(line)

def ensure_file_exists(file_path):
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            pass

def create_or_clear_file(file_path):
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(file_path, 'w', encoding='utf-8'):
        pass

def post_process(opt, dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(),
        [meta["c"]],
        [meta["s"]],
        meta["out_height"],
        meta["out_width"],
        opt.num_classes,
    )
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(opt, detections):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0
        ).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > opt.K:
        kth = len(scores) - opt.K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = results[j][:, 4] >= thresh
            results[j] = results[j][keep_inds]
    return results


def gather_seq_info_multi_view(opt, imgs_dir_path, frame_names, description, aptm, use_cuda=True):
    seq_dict = {}

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    view_detections = defaultdict(list)

    if opt.gpus[0] >= 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(device)
    model.eval()
    
    view = "View1"
    view_ls = ["View1"]

    seq_length = len(frame_names)
    print("seq_length: ", seq_length)

    for frame_idx in range(0, len(frame_names)):
        image_path = os.path.join(imgs_dir_path, frame_names[frame_idx])

        # Read image
        img0 = cv2.imread(image_path)  # BGR
        assert img0 is not None, "Failed to load " + image_path

        def letterbox(
            img, height=608, width=1088, color=(127.5, 127.5, 127.5)
        ):  # resize a rectangular image to a padded rectangular
            shape = img.shape[:2]  # shape = [height, width]
            ratio = min(float(height) / shape[0], float(width) / shape[1])
            new_shape = (
                round(shape[1] * ratio),
                round(shape[0] * ratio),
            )  # new_shape = [width, height]
            dw = (width - new_shape[0]) / 2  # width padding
            dh = (height - new_shape[1]) / 2  # height padding
            top, bottom = round(dh - 0.1), round(dh + 0.1)
            left, right = round(dw - 0.1), round(dw + 0.1)
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )  # padded rectangular
            return img, ratio, dw, dh

        # width = 1088, height = 608
        self_width = opt.img_size[0]
        self_height = opt.img_size[1]

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self_height, width=self_width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        frame_index = frame_idx
        
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = blob.shape[2]
        inp_width = blob.shape[3]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {
            "c": c,
            "s": s,
            "out_height": inp_height // opt.down_ratio,
            "out_width": inp_width // opt.down_ratio,
        }

        # output
        with torch.no_grad():
            output = model(blob)[-1]
            hm = output["hm"].sigmoid_()
            wh = output["wh"]

            if opt.baseline == 0:
                view_id_feature = F.normalize(output["id"], dim=1) # This is cross view id feature
                id_feature = F.normalize(output["single_view_id"], dim=1)
                text_id_feature = F.normalize(output["text_id"], dim=1) # text id feature
            else:
                if opt.baseline_view == 0:
                    id_feature = F.normalize(output["single_view_id"], dim=1)
                else:
                    view_id_feature = F.normalize(output["id"], dim=1) # This is cross view id feature

            reg = output["reg"] if opt.reg_offset else None
            dets, bboxes, scores, clses, inds = mot_decode(
                hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K
            )

            if opt.baseline == 0:
                id_feature = (
                    _tranpose_and_gather_feat(id_feature, inds).squeeze(0).cpu().numpy()
                )
                view_id_feature = (
                    _tranpose_and_gather_feat(view_id_feature, inds)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                text_id_feature = (
                    _tranpose_and_gather_feat(text_id_feature, inds)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            else:
                if opt.baseline_view == 0:
                    id_feature = (
                        _tranpose_and_gather_feat(id_feature, inds)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    view_id_feature = id_feature
                else:
                    view_id_feature = (
                        _tranpose_and_gather_feat(view_id_feature, inds)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    id_feature = view_id_feature
        
        # detections
        dets = post_process(opt, dets, meta)
        dets = merge_outputs(opt, [dets])[1]
        remain_inds = dets[:, 4] > opt.conf_thres
        dets = dets[remain_inds]
        bboxes = bboxes[0][remain_inds]
        scores = scores[0][remain_inds]
        id_feature = id_feature[remain_inds]
        view_id_feature = view_id_feature[remain_inds]

        # ---------------------------------------------------------------------- # 
        text_id_feature = text_id_feature[remain_inds] # Text id feature
        
        if 0 == dets.size:
            # The detection box is empty, exception handling
            scores_attr = []
            scores_text = []
            scores_total = []
        else: 
            # The detection box is not empty
            # APTM
            with torch.no_grad(): # No gradient is required for subsequent calculations
                # Read the original image and convert it to RGB channel order
                original_img = Image.open(image_path).convert('RGB') 

                # The predicted original image coordinate system is x1y1x2y2
                det_bboxs = dets[:,:4] # Select the first four columns

                images = []
                for i, bbox in enumerate(det_bboxs):
                    box = (bbox[0], bbox[1], bbox[2], bbox[3]) # x1y1x2y2
                    region_img = original_img.crop(box) # Image of the required area
                    # region_img.save("test_examples/people" + "_" + str(img_index) + "_" + str(i) + ".jpg")
                    images.append(region_img)
                
                text = description # language description

                CNN_image_features = torch.tensor(text_id_feature).to(device) # Select the corresponding CNN image features and put them on the GPU
                CNN_image_alpha = opt.CNN_image_alpha

                # inference
                scores_attr, scores_text = aptm.inference_calculate_score(text, images, CNN_image_features, CNN_image_alpha)

                # Calculate the total score
                scores_total = opt.score_text_gamma * scores_text + opt.score_attr_beta * np.exp(scores_attr)

        # ---------------------------------------------------------------------- # 
        for feature, view_feature, detection, id, score_attr, score_text, score_total in zip(
            id_feature, view_id_feature, dets, remain_inds, scores_attr, scores_text, scores_total
        ):
            index = frame_index
            confidence = detection[-1]
            detection = [int(i) for i in detection]
            confidence = confidence.item()
            det = (
                [index]
                + [id]
                + [
                    detection[0],
                    detection[1],
                    detection[2] - detection[0],
                    detection[3] - detection[1],
                ]
                + [confidence]
                + [0, 0, 0]
                + [score_attr] # [10]
                + [score_text] # [11]
                + [score_total] # [12]
                + feature.tolist() # [13:] # Adding View Features
            )
            detections[view].append(det)
            view_det = (
                [index]
                + [id]
                + [
                    detection[0],
                    detection[1],
                    detection[2] - detection[0],
                    detection[3] - detection[1],
                ]
                + [confidence]
                + [0, 0, 0]
                + [score_attr] # [10]
                + [score_text] # [11]
                + [score_total] # [12]
                + view_feature.tolist() # [13:] # Adding View Features
            )
            view_detections[view].append(view_det) # Adding View Features

    for view in view_ls:
        view_dict = {
            "image_filenames": frame_names[0],
            "detections": np.array(detections[view]),
            "view_detections": np.array(view_detections[view]),
            "min_frame_idx": 1,
            "max_frame_idx": seq_length,
        }
        seq_dict = view_dict
    return seq_dict


def main(
    opt
):
    logger.setLevel(logging.INFO)

    # input
    testsets_path = "/mnt/A/hust_csj/Code/CRMOT/CRTracker/single_view_demo/data"
    imgs_dir_path = "/mnt/A/hust_csj/Code/CRMOT/CRTracker/single_view_demo/data/imgs"

    # output
    save_results_path = "/mnt/A/hust_csj/Code/CRMOT/CRTracker/single_view_demo/results"
    save_dir_name = "CRTracker_single_view_debug"

    # APTM:
    task = "rstp"
    checkpoint = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_rstp/checkpoint_best.pth"
    config = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_rstp.yaml"
    aptm = APTM(config, task, checkpoint)


    seq_mv = {}
    view_ls = ["View1"]
    view = "View1"

    description_txt = testsets_path + "/description.txt"
    # get language description
    with open(description_txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) >= 2:
            description = lines[1].strip()
        else:
            print("读取语言描述时，发生错误")

    imgs_path_txt = testsets_path + "/imgs_path.txt"
    # get image paths
    with open(imgs_path_txt, 'r', encoding='utf-8') as file:
        frame_names = [os.path.join(line.strip()) for line in file]  

    # save path
    save_results_txt_path = os.path.join(save_results_path, "txt", save_dir_name, "demo.txt")
    ensure_file_exists(save_results_txt_path) 
    create_or_clear_file(save_results_txt_path) 

    for view in view_ls:
        seq_mv[view] = gather_seq_info_multi_view(
            opt, imgs_dir_path, frame_names, description, aptm
        )

    mvtracker = MVTracker(opt, view_ls)

    # display: visualization switch
    updater = Update(
        opt, seq=seq_mv, mvtracker=mvtracker, display=opt.track_display, view_list=view_ls
    )
    updater.run()


    # save results
    for view in view_ls:
        # row[0], row[1], row[2], row[3], row[4], row[5]
        # frame, id, x1, y1, x2, y2
        for row in updater.result[view]: # Get results, write results
            frame_idx, id, x1, y1, x2, y2 = row[0], row[1], row[2], row[3], row[4], row[5]
            image_name = frame_names[frame_idx].split("/")[-1]
            out_box = (x1, y1, x2, y2)
            out_obj_id = id
            write_results(txt_path=save_results_txt_path,
                        frame_name=image_name,
                        bbox_xyxy=out_box,
                        track_id=out_obj_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()

    main(opt)
