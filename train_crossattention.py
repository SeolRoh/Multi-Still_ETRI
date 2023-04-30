import argparse
import random
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader

from models.multimodal_cross_attention import *
from merdataset import *
from config import *
from utils import *
import time

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--epochs',
        default=train_config['epochs'],
        type=int,
        required=False,
        help='epochs'
    )
    parser.add_argument(
        '--batch',
        default=train_config['batch_size'],
        type=int,
        required=False,
        help='batch size'
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        required=False,
        help='shuffle'
    )
    parser.add_argument(
        '--lr',
        default=train_config['lr'],
        type=float,
        required=False,
        help='learning rate'
    )
    parser.add_argument(
        '--cuda',
        default='cuda:0',
        help='class weight'
    )

    parser.add_argument(
        '--save',
        default=True,
        action='store_true',
        help='save checkpoint'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='test',
        help='checkpoint name to load or save'
    )

    parser.add_argument(
        '--text_only',
        type=bool,
        default=False,
        help='train text encoder only'
    )

    parser.add_argument(
        '--audio_only',
        type=bool,
        default=False,
        help='train audio encoder only'
    )

    args = parser.parse_args()
    return args

args = parse_args()
if args.cuda != 'cuda:0':
    audio_config['cuda'] = args.cuda
    text_config['cuda'] = args.cuda
    cross_attention_config['cuda'] = args.cuda
    train_config['cuda'] = args.cuda


def train(model,optimizer, dataloader):
    print("Train start")
    model.train()
    #model.freeze()
    
    # 각 발화 및 스크립트 별 평가자들의 평가결과를 Softmax로 사용, MSEloss를 이용해 학습
    loss_func = torch.nn.MSELoss().to(train_config['cuda'])

    tqdm_train = tqdm(total=len(dataloader), position=1)
    accumulation_steps = train_config['accumulation_steps']
    loss_list = []
    
    for batch_id, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]

        outputs = model(batch_x)
        loss = loss_func(outputs.to(torch.float32).to(train_config['cuda']), batch_y.to(torch.float32).to(train_config['cuda']))
        loss_list.append(loss.item())
        
        tqdm_train.set_description('loss is {:.2f}'.format(loss.item()))
        tqdm_train.update()
        loss = loss / accumulation_steps
        loss.backward()
        if batch_id % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    optimizer.zero_grad()
    tqdm_train.close()
    print("Train Loss: {:.5f}".format(sum(loss_list)/len(loss_list)))

def main():
    audio_conf = pd.Series(audio_config)
    text_conf = pd.Series(text_config)
    cross_attention_conf = pd.Series(cross_attention_config)

    print(audio_conf)
    print(text_conf)
    print(cross_attention_conf)
    print(train_config)

    #audio_conf['path'] = './TOTAL/Extracted_Dataset/'

    # 데이터셋 불러오기
    dataset = MERGEDataset(data_option='train', path='./data/')
    dataset.prepare_text_data(text_conf)

    # 동일한 결과를 얻기위한 seed설정
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 모델 생성
    model = MultiModalForCrossAttention(audio_conf, text_conf, cross_attention_conf, args.text_only, args.audio_only)

    device = args.cuda
    print('---------------------',device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    if 'ckpt' not in os.listdir():
        os.mkdir('ckpt')

    print(model)
    get_params(model)

    if args.save:
        print("checkpoint will be saved every 5epochs!")

    for epoch in range(args.epochs):
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=args.shuffle,
                                    collate_fn=lambda x: (x, torch.FloatTensor([i['label'] for i in x])))
        train(model, optimizer, dataloader)
        
        # 5의 배수 epoch마다 모델 저장
        if (epoch+1) % 5 == 0:
            if args.save:
                torch.save(model,'./ckpt/{}_epoch{}.pt'.format(args.model_name,epoch))



if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total Training time is : ", end_time-start_time)
