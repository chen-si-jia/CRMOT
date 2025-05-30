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


def gather_seq_info_multi_view(opt, dataloader, seq, seq_length, use_cuda=True):
    seq_dict = {}
    # print('loading dataset...')

    image_filenames = defaultdict(list)
    detections = defaultdict(list)
    view_detections = defaultdict(list)
    # model
    # print('Creating model...')
    if opt.gpus[0] >= 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(device)
    model.eval()

    # APTM:
    task = "rstp"
    checkpoint = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_rstp/checkpoint_best.pth"
    config = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_rstp.yaml"
    aptm = APTM(config, task, checkpoint)
    
    view_ls = dataloader.view_list
    for data_i, (path, img, img0) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # # Dedicated for debugging
        # if data_i > 30:
        #     break
        
        if opt.test_divo or opt.test_campus:
            view = path.split("/")[-3].split("_")[1] # Automatically set the view
            frame_index = int(path.split("/")[-1].split(".jpg")[0].split("_")[-1])
        
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
            text_scores =[]
        else: 
            # The detection box is not empty
            # APTM
            with torch.no_grad(): # No gradient is required for subsequent calculations
                # Read the original image and convert it to RGB channel order
                original_img = Image.open(path).convert('RGB') 

                # The predicted original image coordinate system is x1y1x2y2
                det_bboxs = dets[:,:4] # Select the first four columns

                # Drawing Test
                if False:
                    img0 = cv2.imread(path)
                    # xyxy Picture Frame
                    for i in range(len(det_bboxs)):
                        cv2.rectangle(img0, (int(det_bboxs[i][0]), int(det_bboxs[i][1])), (int(det_bboxs[i][2]), int(det_bboxs[i][3])), (0, 0, 255), 2)
                    cv2.imwrite("img" + "_" + str(img_index) + ".jpg", img0) 
                    # Get the image of each person in the picture
                    people_imgs = []
                    for i, bbox in enumerate(det_bboxs):
                        box = (bbox[0], bbox[1], bbox[2], bbox[3]) # x1y1x2y2
                        region = original_img.crop(box) 
                        people_imgs.append(region)
                        region.save("people" + "_" + str(img_index) + "_" + str(i) + ".jpg")

                images = []
                for i, bbox in enumerate(det_bboxs):
                    box = (bbox[0], bbox[1], bbox[2], bbox[3]) # x1y1x2y2
                    region_img = original_img.crop(box) # Image of the required area
                    # region_img.save("test_examples/people" + "_" + str(img_index) + "_" + str(i) + ".jpg")
                    images.append(region_img)

                if "text_prompt_4" == opt.text_prompt:
                    # eg: a man in a white coat and black trousers
                    text = ["a" + seq.split("_")[-1].split(":")[0].split('A')[1].split('.')[0],] # input text，List
                elif "text_prompt_5" == opt.text_prompt:
                    # eg: A man in a white coat and black trousers.
                    text = ["a" + seq.split("_")[-1].split(":")[0],] # input text，List
                else:
                    assert("text_prompt input error")

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
            "image_filenames": seq,
            "detections": np.array(detections[view]),
            "view_detections": np.array(view_detections[view]),
            "image_size": (3, 1920, 1080),
            "min_frame_idx": 1,
            "max_frame_idx": seq_length,
        }
        if opt.test_divo or opt.test_campus:
            seq_dict = view_dict
        if opt.test_mvmhat or opt.test_mvmhat_campus or opt.test_wildtrack:
            view_dict["min_frame_idx"] = int(seq_length * 2 / 3) + 1
            seq_dict[view] = view_dict
        if opt.test_epfl:
            view_dict["min_frame_idx"] = int(seq_length)
            view_dict["max_frame_idx"] = int(seq_length * 3 / 2)
    return seq_dict


def main(
    opt,
    data_root="/data/MOT16/train",
    seqs=("MOT16-05",),
    exp_name="demo",
):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, "..", "results", exp_name)
    mkdir_if_missing(result_root)
    view_ls = []
    
    # run tracking
    for seq in seqs:
        logger.info("start seq: {}".format(seq))
        if opt.test_divo:
            scene = seq.split("_")[0]
            train_ls = ["Shop", "Floor", "Gate1", "Park", "Ground", "Moving", "Square"]
            test_ls = ["Gate2", "Circle", "Side"]
            view_ls = ["View1", "View2", "View3"]

            if scene in test_ls:
                seq_mv = {}
                for view in view_ls:
                    dataloader = datasets.LoadImages_DIVO(
                        opt,
                        osp.join("/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/test", "{}_{}".format(seq.split("_")[0], view), "img1"), 
                        opt.img_size,
                    )
                    seq_mv[view] = gather_seq_info_multi_view(
                        opt, dataloader, seq, dataloader.seq_length
                    )
            if scene in train_ls:
                seq_mv = {}
                for view in view_ls:
                    dataloader = datasets.LoadImages_DIVO(
                        opt,
                        osp.join("/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/train", "{}_{}".format(seq.split("_")[0], view), "img1"), 
                        opt.img_size,
                    )
                    seq_mv[view] = gather_seq_info_multi_view(
                        opt, dataloader, seq, dataloader.seq_length
                    )
        elif opt.test_campus:
            scene = seq.split("_")[0]
            test_ls = ["Garden1", "Garden2", "ParkingLot"]
            # Set the number of views according to the scene
            if ("Garden1" == scene) or ("ParkingLot" == scene):
                view_ls = ["View1", "View2", "View3", "View4"]
            else:
                view_ls = ["View1", "View2", "View3"]

            if scene in test_ls:
                seq_mv = {}
                for view in view_ls:
                    dataloader = datasets.LoadImages_DIVO(
                        opt,
                        osp.join("/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_Cross-domain/images/test", "{}_{}".format(seq.split("_")[0], view), "img1"), 
                        opt.img_size,
                    )
                    seq_mv[view] = gather_seq_info_multi_view(
                        opt, dataloader, seq, dataloader.seq_length
                    )
        else:
            dataloader = datasets.LoadImages(
                opt, osp.join(data_root, seq), opt.img_size
            )
            meta_info = open(os.path.join(data_root, seq, "seqinfo.ini")).read()
            seq_length = int(
                meta_info[
                    meta_info.find("seqLength=") + 10 : meta_info.find("\nimWidth")
                ]
            )
            seq_mv = gather_seq_info_multi_view(opt, dataloader, seq, seq_length)
            view_ls = dataloader.view_list

        mvtracker = MVTracker(opt, view_ls)

        # display: visualization switch
        updater = Update(
            opt, seq=seq_mv, mvtracker=mvtracker, display=opt.track_display, view_list=view_ls
        )
        updater.run()

        for view in view_ls:
            if not os.path.exists(os.path.join(result_root, seq)):
                os.mkdir(os.path.join(result_root, seq))
            f = open(os.path.join(result_root, seq, "{}.txt".format(view)), "w")
            for row in updater.result[view]: # Get results, write results
                print(
                    "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                    % (row[0], row[1], row[2], row[3], row[4], row[5]),
                    file=f,
                )
            f.close()

            # single view visualization:
            # row[0], row[1], row[2], row[3], row[4], row[5]
            # frame, id, x1, y1, x2, y2
            # if opt.vis and online_tlwhs:
            #     plot_mot_tracking_online_add_conf(img0, online_tlwhs, online_ids, online_scores, frame_id=frame_id, seq_name=seq, opt=opt)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = opts().init()

    # inference on In-domain test set(the DIVOTrack dataset).
    if opt.test_divo:
        seqs_str = """Gate2_A man wearing a white coat, black trousers and white shoes.:5
                    Gate2_A man wearing a black coat and black trousers.:1,2
                    Gate2_A man wearing a black coat, black trousers and white shoes.:2
                    Gate2_A man wearing a black coat, gray trousers and gray shoes.:3
                    Gate2_A man riding a bicycle.:4
                    Circle_A man wearing a white coat and black trousers.:1,27,30
                    Circle_A man wearing a white coat, black trousers and white shoes.:1
                    Circle_A man wearing a white coat, gray trousers and black shoes.:19
                    Circle_A man wearing a black coat and black trousers.:4,6,8,9,14,15,23,24,26,29,40
                    Circle_A man wearing a black cap, black coat and black trousers.:14,45
                    Circle_A man wearing a black coat, gray trousers and gray shoes.:3,43
                    Circle_A man wearing a black coat, gray trousers and white shoes.:44
                    Circle_A man wearing a black coat, black trousers and black shoes.:6,9,14,15,23,26,40
                    Circle_A man wearing a black coat, black trousers and white shoes.:8,24,29
                    Circle_A man in a black coat and blue trousers, dragging a cart.:16
                    Circle_A man wearing a gray coat and black trousers.:28,41
                    Circle_A man wearing a red coat.:25,34
                    Circle_A man wearing a red coat and black trousers.:34
                    Circle_A man wearing a yellow coat.:32,42,46
                    Circle_A man wearing a yellow coat and gray trousers.:42
                    Circle_A man wearing a yellow coat, black trousers and red shoes.:32
                    Circle_A man wearing a blue coat, black trousers and black shoes.:22
                    Circle_A man wearing a blue helmet, blue coat and black trousers.:18,31,33
                    Circle_A man in a blue helmet, blue coat, black trousers and black shoes, carrying a red plastic bag.:18
                    Circle_A man in a blue coat, black trousers and white shoes, carrying a white plastic bag.:47
                    Side_A man wearing a white coat.:2,4,27,28,41,54
                    Side_A man wearing a white coat, black trousers and white shoes.:2,4,28,54
                    Side_A man wearing a black coat.:3,9,25,29,32,40,43,45,48,52,53
                    Side_A man wearing a black coat and black trousers.:3,25,32,52
                    Side_A man wearing a gray coat.:8,14,16,18,20,26,36
                    Side_A man wearing a gray coat and white trousers.:16
                    Side_A man wearing a gray coat and black trousers.:20,26
                    Side_A man in a gray coat, carrying a white bag.:18
                    Side_A man wearing a green coat.:15,23
                    Side_A man wearing a green coat and black trousers.:15,23
                    Side_A man wearing a pink coat.:22,24,30,35,49
                    Side_A man wearing a pink coat and gray trousers.:30
                    Side_A man wearing a red coat.:10,46,47
                    Side_A man wearing a yellow coat.:11,19,21,31,33
                    Side_A man wearing a yellow coat and gray trousers.:31,33"""

        # The results folder serves as a guide and cannot be deleted.
        data_root = os.path.join(opt.data_dir, "CRTrack_In-domain/labels_with_ids_text/test/results")

    # inference on Cross-domain test set (the campus dataset).
    if opt.test_campus:
        seqs_str = """Garden1_A man wearing a white coat and pink trousers.:0
                    Garden1_A man wearing a white coat and gray trousers.:4
                    Garden1_A man wearing a black coat and blue trousers.:10
                    Garden1_A man wearing a black coat and gary trousers.:2,13,15
                    Garden1_A man in a white cap, black coat and gray trousers, holding a stick.:2
                    Garden1_A man in a black helmet, black and white coat and gray trousers, riding a bicycle.:13
                    Garden1_A man wearing a gray coat and black trousers.:11
                    Garden1_A man wearing a green coat and black trousers.:5
                    Garden1_A man wearing a red coat and blue trousers.:8
                    Garden1_A man wearing a red coat and black trousers.:6,14
                    Garden1_A man wearing a red cap, red coat and black trousers.:6
                    Garden1_A man wearing a red and purple coat and blue trousers.:3
                    Garden1_A man in a yellow coat and gray trousers, holding a dog.:1
                    Garden1_A man wearing a blue coat and black trousers.:7
                    Garden1_A man wearing a blue coat and blue trousers.:9
                    Garden1_A man in a black helmet, blue coat and gray trousers, riding a bicycle.:12
                    Garden2_A man wearing a white coat and black trousers.:3,17
                    Garden2_A man in a white coat and black trousers, carrying a black and blue schoolbag.:3
                    Garden2_A man wearing a black coat and blue trousers.:1,2,5,7,9,10,13,16
                    Garden2_A man wearing a black coat and gray trousers.:8
                    Garden2_A man in a cap, black coat and blue trousers, holding a white dog.:1
                    Garden2_A man in a black coat and blue trousers, carrying a black schoolbag.:2
                    Garden2_A man in a black coat and blue trousers, carrying a blue bag and holding a book.:10
                    Garden2_A man in a black coat and blue trousers, carrying a mobile phone.:13
                    Garden2_A man in a black and green coat and blue trousers, holding a book.:16
                    Garden2_A man in a red and black coat and blue trousers, holding a book.:7
                    Garden2_A man in a gray coat and black trousers, holding a dog.:4
                    Garden2_A man in a black helmet, gray coat and black trousers, riding a bicycle.:0
                    Garden2_A man wearing a red coat and black trousers.:6
                    Garden2_A man in a yellow coat and blue trousers, carrying a schoolbag.:11
                    Garden2_A man wearing a yellow cap, orange and blue coat and gary trousers.:15
                    Garden2_A man in a blue coat and gray trousers, carrying a black schoolbag.:12
                    Garden2_A man wearing an orange and gary coat and gary trousers.:14
                    ParkingLot_A man, carrying a schoolbag and holding a red can.:2
                    ParkingLot_A man in a white coat, riding a bicycle.:1
                    ParkingLot_A man wearing a white coat and black trousers.:4
                    ParkingLot_A man wearing a black coat and blue trousers.:5,11
                    ParkingLot_A man wearing a black coat and white trousers.:7
                    ParkingLot_A man in a black coat and blue trousers, holding a dog.:11
                    ParkingLot_A man wearing a green coat and black trousers.:6
                    ParkingLot_A man in a green coat, carrying a bag.:10
                    ParkingLot_A man in a green  coat, riding a bicycle.:13
                    ParkingLot_A man wearing a red coat and blue trousers.:9
                    ParkingLot_A man wearing a blue coat.:0,3,8
                    ParkingLot_A man wearing a blue coat and gray trousers.:0
                    ParkingLot_A man wearing a blue coat and blue trousers.:3
                    ParkingLot_A man wearing a blue coat and black trousers.:14
                    ParkingLot_A man in a cap, purple coat and gray trousers, carrying a schoolbag.:12"""
        
        # The results folder serves as a guide and cannot be deleted.
        data_root = os.path.join(opt.data_dir, "CRTrack_Cross-domain/labels_with_ids_text/test/results")
    
    if opt.test_divo or opt.test_campus:
        seqs = [seq.strip() for seq in seqs_str.split("\n")]
    else:
        seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt, data_root=data_root, seqs=seqs, exp_name=opt.exp_name)
