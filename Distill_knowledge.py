import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from models.multimodal import TextEncoder, SpeechEncoder
from merdataset import *
from mini_config import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--cuda',
        default=test_config['cuda'],
        help='cuda'
    )

    parser.add_argument(
        '--teacher_name',
        type=str,
        help='checkpoint name to load'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default="data/train_preprocessed_data.json",
        help="Distilled teacher's knowledge path"
    )

    args = parser.parse_args()
    return args

args = parse_args()
if args.cuda != 'cuda:0':
    text_config['cuda'] = args.cuda
    test_config['cuda'] = args.cuda

    
def main():
    text_conf = pd.Series(text_config)
    if args.teacher_name:
        
        # 교사 모델 불러오기
        model = torch.load('./ckpt/{}.pt'.format(args.teacher_name))
        model.eval()
        #print(model)
    
        # 데이터 파일 읽기
        with open(args.data_path,'r') as file:
            json_data = json.load(file)
        
        # 데이터의 'knoledge_distillation'을 키로, 교사모델의 예측 softmax값 추가
        for i in range(len(json_data['data'])):
            if json_data['data'][i].get('knoledge_distillation', 1):
                K = text_config['K']
                dialogue = json_data['data'][i]['utterance']
                json_data['data'][i]['dialogue'] = dialogue
                output = model([json_data['data'][i]])
                json_data['data'][i]['knoledge_distillation'] = output.tolist()
        with open(args.data_path,'w') as j:
            json.dump(json_data,j,ensure_ascii=False, indent=4)

    else:
        print("You need to define specific model name to test")


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
