import os.path as osp
import os
from tqdm import tqdm

root = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/'
dataset = 'CRTrack_In-domain/images/train'
label = '/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain/labels_with_ids/train'
train_file = '../data/CRTrack_In-domain.train'


seqs = os.listdir(osp.join(root, dataset))
seqs.sort()

with open(train_file, 'a') as f:
    for seq in seqs:
        path = osp.join(root, dataset, seq)
        file_list = os.listdir(path)
        file_list.sort()

        for filename in file_list:
            if (filename.split('.')[-1] == 'ini'):
                seq_info = open(osp.join(path, filename)).read()

        seq_length = int(int(seq_info[seq_info.find('seqLength=') + 10:seq_info.find('\nimWidth')]))

        img_path = osp.join(path, 'img1')
        img_list = sorted(os.listdir(img_path))

        for filename in tqdm(img_list):
            if (filename.split('.')[-1] == 'jpg'):
                name = filename.split('.')[0]

                if filename.split('.')[0]+'.txt' in os.listdir(osp.join(label, seq, 'img1')):
                    f.write(osp.join(dataset, seq, 'img1', filename) + '\n')