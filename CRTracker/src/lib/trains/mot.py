from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer

from PIL import Image
from torchvision import transforms
import cv2
from tracker.multitracker import JDETracker_to_bbox
import os

from aptm.aptm_module import APTM


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = (
            RegL1Loss()
            if opt.reg_loss == "l1"
            else RegLoss()
            if opt.reg_loss == "sl1"
            else None
        )
        self.crit_wh = (
            torch.nn.L1Loss(reduction="sum")
            if opt.dense_wh
            else NormRegL1Loss()
            if opt.norm_wh
            else RegWeightedL1Loss()
            if opt.cat_spec_wh
            else self.crit_reg
        )
        self.opt = opt
        self.baseline = self.opt.baseline
        self.baseline_view = self.opt.baseline_view
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.view_nID = opt.view_nID
        if self.baseline == 0:
            self.classifier = nn.Linear(int(self.emb_dim), self.nID)
            self.view_classifier = nn.Linear(int(self.emb_dim), self.view_nID)
        else:
            if self.baseline_view == 0:
                self.view_classifier = nn.Linear(int(self.emb_dim), self.view_nID)
            else:
                self.classifier = nn.Linear(int(self.emb_dim), self.nID)
        
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.zero = torch.tensor(0.0).cuda()
        self.single_loss_array = opt.single_loss_array
        self.single_view_id_split_loss = opt.single_view_id_split_loss
        self.cross_loss_array = opt.cross_loss_array
        self.cross_view_id_split_loss = opt.cross_view_id_split_loss

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Read and process text tags
        label_with_ids_text_Path = "/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_text/train/gt_train"
        self.all_texts = []
        self.id_text = []
        files = os.listdir(label_with_ids_text_Path)

        if "text_prompt_1" == opt.text_prompt:
            # Original sentence without the period
            for file in files:
                txtFile = open(os.path.join(label_with_ids_text_Path, file),'r')
                for line in txtFile.readlines():    
                    temp = line.strip()
                    data = temp.split(':') 
                    id = data[0]
                    text = data[1].split('.')[0] 
                    if text not in self.all_texts:
                        self.all_texts.append(text)
                    self.id_text.append([file.split('.')[0]+"_"+id, text])
        elif "text_prompt_2" == opt.text_prompt:
            # A photo of + Original sentence without the period
            for file in files:
                txtFile = open(os.path.join(label_with_ids_text_Path, file),'r')
                for line in txtFile.readlines():
                    temp = line.strip()
                    data = temp.split(':')
                    id = data[0]
                    text = "A photo of" + data[1].split('A')[1] 
                    if text not in self.all_texts:
                        self.all_texts.append(text)
                    self.id_text.append([file.split('.')[0]+"_"+id, text])
        elif "text_prompt_3" == opt.text_prompt:
            # A photo of + The sentence where the beginning of the original sentence A is replaced by a
            for file in files:
                txtFile = open(os.path.join(label_with_ids_text_Path, file),'r')
                for line in txtFile.readlines():
                    temp = line.strip()
                    data = temp.split(':')
                    id = data[0]
                    text = "A photo of" + " a" + data[1].split('A')[1] 
                    if text not in self.all_texts:
                        self.all_texts.append(text)
                    self.id_text.append([file.split('.')[0]+"_"+id, text])
        elif "text_prompt_4" == opt.text_prompt:
            # The sentence where the beginning of the original sentence is replaced by a and the period at the end is removed
            # eg: a man in a white coat and black trousers
            for file in files:
                txtFile = open(os.path.join(label_with_ids_text_Path, file),'r')
                for line in txtFile.readlines():
                    temp = line.strip()
                    data = temp.split(':')
                    id = data[0]
                    text = "a" + data[1].split('A')[1].split('.')[0]
                    if text not in self.all_texts:
                        self.all_texts.append(text)
                    self.id_text.append([file.split('.')[0]+"_"+id, text])
        elif "text_prompt_5" == opt.text_prompt:
            # Original sentence
            # eg: A man in a white coat and black trousers.
            for file in files:
                txtFile = open(os.path.join(label_with_ids_text_Path, file),'r')
                for line in txtFile.readlines():    
                    temp = line.strip()
                    data = temp.split(':')
                    id = data[0]
                    text = data[1]
                    if text not in self.all_texts:
                        self.all_texts.append(text)
                    self.id_text.append([file.split('.')[0]+"_"+id, text])
        else:
            while 1:
                print("text_mean input error")

        # Add an empty text at the end as an ID without a tag
        self.all_texts.append("") 
        # save all texts
        with open("all_texts.txt", "w") as file:
            for text in self.all_texts:
                file.write(text+"\n")
        
        # APTM:
        task = "rstp"
        checkpoint = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_rstp/checkpoint_best.pth"
        config = "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_rstp.yaml"
        self.aptm = APTM(config, task, checkpoint)

        self.tracker_to_bbox = JDETracker_to_bbox(opt)


    def forward(self, outputs, batch): 
        # outputs：Model Output
        # batch：gt label
        # hm： Category, the heatmap value is a probability value between 0 and 1.
        #      If there is an object of a certain category in the image, 
        #      the probability value of the center point of this object on the heatmap is 1.
        # wh： Target width and height
        # reg：Target center offset xy
        # id： Target ID
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        text_id_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output["hm"] = _sigmoid(output["hm"])

            hm_loss += self.crit(output["hm"], batch["hm"]) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += (
                    self.crit_reg(
                        output["wh"], batch["reg_mask"], batch["ind"], batch["wh"]
                    )
                    / opt.num_stacks
                )

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += (
                    self.crit_reg(
                        output["reg"], batch["reg_mask"], batch["ind"], batch["reg"]
                    )
                    / opt.num_stacks
                )

            if opt.id_weight > 0:
                if self.baseline == 0:
                    id_head = _tranpose_and_gather_feat(
                        output["id"], batch["ind"]
                    )
                    id_head = id_head[batch["reg_mask"] > 0].contiguous()
                    id_head = self.emb_scale * F.normalize(id_head)
                    id_target = batch["ids"][batch["reg_mask"] > 0]
                    id_output = self.classifier(id_head).contiguous()

                    text_id_head = _tranpose_and_gather_feat(
                        output["text_id"], batch["ind"]
                    )
                    text_id_head = text_id_head[batch["reg_mask"] > 0].contiguous()
                    text_id_head = self.emb_scale * F.normalize(text_id_head)
                    text_id_target2 = []
                    for j in range(len(batch["text_ids"][0])):
                        for i in range(len(batch["text_ids"])):
                            if "" != batch["text_ids"][i][j]:
                                text_id_target2.append(batch["text_ids"][i][j])
                            else:
                                continue # When encountering "", end the column directly
                    text_id_targets = []

                    for i, text_id_value1 in enumerate(text_id_target2):
                        for j, id_text in enumerate(self.id_text):
                            if id_text[0] == text_id_value1:
                                index = self.all_texts.index(id_text[1]) # Subscripts start at 0
                                text_id_targets.append(index)
                                break
                            # If the traversal is completed and no corresponding text is found, the subscript corresponding to the empty text is set.
                            if j == len(self.id_text) - 1:
                                text_id_targets.append(len(self.all_texts) - 1)
                    
                    text_id_targets = torch.LongTensor(text_id_targets).to(self.device) # Convert list to tensor int64 and put it on GPU

                    ################################################################
                    # APTM:
                    # Using APTM to process gt frames
                    for img_index, img_path in enumerate(batch["img_path"]):
                        original_img = Image.open(img_path).convert('RGB') # Read the original image and convert it to RGB channel order
                        (w, h) = original_img.size

                        # gt data -> x1y1x2y2 in gt's original image coordinate system
                        gt_bboxs = self.tracker_to_bbox.gt_to_bbox(w, h, batch, img_index)

                        images = []
                        for i, bbox in enumerate(gt_bboxs):
                            box = (bbox[0], bbox[1], bbox[2], bbox[3]) # x1y1x2y2
                            region_img = original_img.crop(box) # Image of the required area
                            # region_img.save("test_examples/people" + "_" + str(img_index) + "_" + str(i) + ".jpg")
                            images.append(region_img)
                        
                        texts = self.all_texts
                        # Statistical interval
                        start = 0
                        for i in range(img_index):
                            start += sum(batch['reg_mask'][i] > 0)
                        end = start + sum(batch['reg_mask'][img_index] > 0)

                        CNN_image_features = text_id_head[start:end] # Select the corresponding CNN image features
                        need_text_id_targets = text_id_targets[start:end] # Select the corresponding text
                        CNN_image_alpha = self.opt.CNN_image_alpha

                        # train
                        similarity = self.aptm.train(texts, images, CNN_image_features, CNN_image_alpha)

                        # calculate loss
                        text_id_loss += self.IDLoss(similarity, need_text_id_targets)
                    ################################################################

                    single_view_id_head = _tranpose_and_gather_feat(
                        output["single_view_id"], batch["ind"]
                    )
                    single_view_id_head = single_view_id_head[
                        batch["reg_mask"] > 0
                    ].contiguous()
                    single_view_id_head = self.emb_scale * F.normalize(
                        single_view_id_head
                    )
                    single_view_id_target = batch["single_view_ids"][
                        batch["reg_mask"] > 0
                    ]
                    single_view_id_output = self.view_classifier(
                        single_view_id_head
                    ).contiguous()
                else:
                    if self.baseline_view == 0:
                        single_view_id_head = _tranpose_and_gather_feat(
                            output["single_view_id"], batch["ind"]
                        )
                        single_view_id_head = single_view_id_head[
                            batch["reg_mask"] > 0
                        ].contiguous()
                        single_view_id_head = self.emb_scale * F.normalize(
                            single_view_id_head
                        )
                        single_view_id_target = batch["single_view_ids"][
                            batch["reg_mask"] > 0
                        ]
                        single_view_id_output = self.view_classifier(
                            single_view_id_head
                        ).contiguous()
                    else:
                        id_head = _tranpose_and_gather_feat(
                            output["id"], batch["ind"]
                        )
                        id_head = id_head[batch["reg_mask"] > 0].contiguous()
                        id_head = self.emb_scale * F.normalize(id_head)
                        id_target = batch["ids"][batch["reg_mask"] > 0]
                        id_output = self.classifier(id_head).contiguous()

                if self.opt.id_loss == "focal":
                    id_target_one_hot = id_output.new_zeros(
                        (id_head.size(0), self.nID)
                    ).scatter_(1, id_target.long().view(-1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(
                        id_output,
                        id_target_one_hot,
                        alpha=0.25,
                        gamma=2.0,
                        reduction="sum",
                    ) / id_output.size(0)
                else:
                    if self.baseline == 0:
                        if len(single_view_id_target) > 0:
                            if self.single_view_id_split_loss:
                                single_id_loss = 0
                                single_loop_target = single_view_id_target
                                single_loop_output = single_view_id_output
                                single_view_num = 0
                                while len(single_loop_target) > 0:
                                    single_view_num += 1
                                    sample_id = single_loop_target[0]
                                    small_id, big_id = 0, 0
                                    for i in range(len(self.single_loss_array)):
                                        if sample_id > self.single_loss_array[i]:
                                            continue
                                        else:
                                            small_id = (
                                                self.single_loss_array[i - 1]
                                                if i != 0
                                                else 0
                                            )
                                            big_id = self.single_loss_array[i]
                                            break
                                    temp_single_output = single_loop_output
                                    temp_single_target = single_loop_target
                                    for i in range(len(single_loop_target)):
                                        if (
                                            single_loop_target[i] <= small_id
                                            or single_loop_target[i] > big_id
                                        ):
                                            temp_single_output = single_loop_output[
                                                :i
                                            ].clone()
                                            temp_single_target = single_loop_target[
                                                :i
                                            ].clone()
                                            single_loop_target = single_loop_target[
                                                i:
                                            ].clone()
                                            single_loop_output = single_loop_output[
                                                i:
                                            ].clone()
                                            break
                                        else:
                                            if i == len(single_loop_target) - 1:
                                                temp_single_output = single_loop_output
                                                temp_single_target = single_loop_target
                                                single_loop_target = []
                                    temp_single_output = temp_single_output[
                                        :, small_id:big_id
                                    ].clone()
                                    temp_single_target[::] = temp_single_target[
                                        ::
                                    ].clone() - (small_id + 1)
                                    single_id_loss += self.IDLoss(
                                        temp_single_output, temp_single_target
                                    )
                                single_id_loss /= single_view_num
                            else:
                                single_id_loss = self.IDLoss(
                                    single_view_id_output, single_view_id_target
                                )
                        else:
                            single_id_loss = self.zero

                        if len(id_target) > 0:
                            if self.cross_view_id_split_loss:
                                cross_id_loss = 0
                                cross_loop_target = id_target
                                cross_loop_output = id_output
                                cross_view_num = 0
                                while len(cross_loop_target) > 0:
                                    cross_view_num += 1
                                    sample_id = cross_loop_target[0]
                                    small_id, big_id = 0, 0
                                    for i in range(len(self.cross_loss_array)):
                                        if sample_id > self.cross_loss_array[i]:
                                            continue
                                        else:
                                            small_id = self.cross_loss_array[i - 1]
                                            big_id = self.cross_loss_array[i]
                                            break
                                    temp_cross_output = cross_loop_output
                                    temp_cross_target = cross_loop_target
                                    for i in range(len(cross_loop_target)):
                                        if (
                                            cross_loop_target[i] <= small_id
                                            or cross_loop_target[i] > big_id
                                        ):
                                            temp_cross_output = cross_loop_output[
                                                :i
                                            ].clone()
                                            temp_cross_target = cross_loop_target[
                                                :i
                                            ].clone()
                                            cross_loop_target = cross_loop_target[
                                                i:
                                            ].clone()
                                            cross_loop_output = cross_loop_output[
                                                i:
                                            ].clone()
                                            break
                                        else:
                                            if i == len(cross_loop_target) - 1:
                                                temp_cross_output = cross_loop_output
                                                temp_cross_target = cross_loop_target
                                                cross_loop_target = []
                                    temp_cross_output = temp_cross_output[
                                        :, small_id:big_id
                                    ].clone()
                                    temp_cross_target[::] = temp_cross_target[
                                        ::
                                    ].clone() - (small_id + 1)
                                    cross_id_loss += self.IDLoss(
                                        temp_cross_output, temp_cross_target
                                    )
                                cross_id_loss /= cross_view_num
                            else:
                                cross_id_loss = self.IDLoss(id_output, id_target)
                        else:
                            cross_id_loss = self.zero

                        id_loss = single_id_loss + cross_id_loss
                    else:
                        if self.baseline_view == 0:
                            single_id_loss = self.IDLoss(
                                single_view_id_output, single_view_id_target
                            )
                            id_loss = single_id_loss
                        else:
                            cross_id_loss = self.IDLoss(id_output, id_target)
                            id_loss = cross_id_loss

        # loss
        det_loss = (
            opt.hm_weight * hm_loss
            + opt.wh_weight * wh_loss
            + opt.off_weight * off_loss
            if id_loss != 0.0
            else self.zero
        )
        if opt.multi_loss == "uncertainty":
            loss = (
                torch.exp(-self.s_det) * det_loss
                + torch.exp(-self.s_id) * id_loss
                + (self.s_det + self.s_id)
            )
            loss *= 0.5
            loss = loss + self.opt.weight_text_id_loss * text_id_loss
        else:
            loss = det_loss + 0.1 * id_loss + self.opt.weight_text_id_loss * text_id_loss

        if self.baseline == 0:
            loss_states = {
                "loss": loss,
                "hm_loss": hm_loss,
                "wh_loss": wh_loss,
                "off_loss": off_loss,
                "det_loss": det_loss,
                "id_loss": id_loss,
                "single_id_loss": single_id_loss,
                "cross_id_loss": cross_id_loss,
                "text_id_loss": text_id_loss,
            }
        else:
            if self.baseline_view == 0:
                loss_states = {
                    "loss": loss,
                    "hm_loss": hm_loss,
                    "wh_loss": wh_loss,
                    "off_loss": off_loss,
                    "single_id_loss": id_loss,
                }
            else:
                loss_states = {
                    "loss": loss,
                    "hm_loss": hm_loss,
                    "wh_loss": wh_loss,
                    "off_loss": off_loss,
                    "cross_id_loss": id_loss,
                }
        
        return loss, loss_states


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.baseline == 0:
            loss_states = [
                "loss",
                "hm_loss",
                "wh_loss",
                "off_loss",
                "det_loss",
                "id_loss",
                "single_id_loss",
                "cross_id_loss",
                "text_id_loss",
            ]
        else:
            if opt.baseline_view == 0:
                loss_states = [
                    "loss",
                    "hm_loss",
                    "wh_loss",
                    "off_loss",
                    "single_id_loss",
                ]
            else:
                loss_states = [
                    "loss",
                    "hm_loss",
                    "wh_loss",
                    "off_loss",
                    "cross_id_loss",
                ]
        loss = MotLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output["reg"] if self.opt.reg_offset else None
        dets = mot_decode(
            output["hm"],
            output["wh"],
            reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh,
            K=self.opt.K,
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(),
            batch["meta"]["c"].cpu().numpy(),
            batch["meta"]["s"].cpu().numpy(),
            output["hm"].shape[2],
            output["hm"].shape[3],
            output["hm"].shape[1],
        )
        results[batch["meta"]["img_id"].cpu().numpy()[0]] = dets_out[0]
