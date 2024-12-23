import os.path as osp
import os
import numpy as np
import pdb
from tqdm import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

ori_label_root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids/train'
tar_label_root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids_cross_view/train'

mkdirs(tar_label_root)
# seqs = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']

# train set：
seqs = ['innerShop', 'movingView', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideSquare', 'park']

seqs_dict = {'circleRegion': 'Circle',
             'innerShop': 'Shop',
             'movingView': 'Moving',
             'park': 'Park',
             'playground': 'Ground',
             'shopFrontGate': 'Gate1',
             'shopSecondFloor': 'Floor',
             'shopSideGate': 'Side',
             'shopSideSquare': 'Square',
             'southGate': 'Gate2'}

views = ['View1', 'View2', 'View3']


min_frame = 0
base_person_id = 0
max_person_id = 0

for seq in seqs:
    for view in views:
        ori_txt_path = osp.join(ori_label_root, seqs_dict[seq] + "_" + view, "img1")
        ori_txt_list = sorted(os.listdir(ori_txt_path))
        for i in ori_txt_list:
            if not '.txt' in i:
                ori_txt_list.remove(i)
        min_frame = int(ori_txt_list[0].split('_')[-1].split('.')[0])

        seq_label_root = osp.join(tar_label_root, seqs_dict[seq] + "_" + view)

        mkdirs(seq_label_root)
        seq_label_root = osp.join(seq_label_root, 'img1')
        mkdirs(seq_label_root)
        for ori_txt_file in tqdm(ori_txt_list):
            if view in ori_txt_file and '.txt' in ori_txt_file:
                txt = osp.join(ori_txt_path, ori_txt_file)
                gt = np.genfromtxt(txt, dtype=np.float64, delimiter=' ')
                if gt.size <= 6:
                    gt = gt.reshape(1, 6)
                cur_frame = int(ori_txt_file.split('_')[-1].split('.')[0])
                for fid, tid, a, b, c, d in gt:
                    label_fpath = osp.join(seq_label_root, '{}_{}_{:06d}.txt'.format(seqs_dict[seq], view, cur_frame - min_frame + 1))
                    label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(int(tid), a, b, c, d)
                    with open(label_fpath, 'a') as f:
                        f.write(label_str)
