import os.path as osp
import os
import numpy as np
from tqdm import tqdm

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/train'
label_root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids/train'
gt_root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/images/train'

mkdirs(label_root)
# seqs = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']

# train sets:
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

tid_curr = 0
tid_last = -1
for seq in tqdm(seqs):
    for view in views:
        seq_info = open(osp.join(seq_root, seqs_dict[seq] + "_" + view, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        seq_label_root = osp.join(label_root, seqs_dict[seq] + "_" + view)
        mkdirs(seq_label_root)

        seq_label_root = osp.join(seq_label_root, 'img1')
        mkdirs(seq_label_root)

        gt_txt = osp.join(gt_root, seqs_dict[seq] + "_" + view,"gt", 'gt.txt')
        gt = np.genfromtxt(gt_txt, dtype=np.float64, delimiter=',')
        
        # gt (MOT17 format)：x_min, y_min, w, h
        for fid, tid, x, y, w, h, _, _, _ in gt:
            fid = int(fid)
            tid = int(tid)
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{}_{}_{:06d}.txt'.format(seqs_dict[seq], view, fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)      
