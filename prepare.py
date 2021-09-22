import numpy as np
import os
from tqdm import tqdm
import argparse


def prepare_data(src_path, out_path):

    cls_paths = [src_path + '/' + str(i) for i in range(cls_num)]

    fo = open(out_path, 'w')
    for idx, cls_path in enumerate(tqdm(cls_paths)):
        files = os.listdir(cls_path)
        for file in files:
            image_path = cls_path + os.sep + file
            info = image_path + '\t' + str(idx) + '\n'
            fo.write(info)
    fo.close()


def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--src',
                        help='path to src data',
                        required=True,
                        type=str)
    parser.add_argument('--out',
                    help='path to save txt',
                    required=True,
                    type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    ''' e.g.: python prepare.py --src ./data/food/train --out ./data/food/train.txt '''
    '''     : python prepare.py --src ./data/food/val --out ./data/food/val.txt     '''
    
    args = parse_args()
    cls_num = 101

    src_path = args.src
    out_path = args.out
    
    prepare_data(src_path, out_path)
    