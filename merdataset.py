import os
import json
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import *


class MERGEDataset(Dataset):
    def __init__(self,data_option='train',path='./'):
        # train, test 데이터셋 읽기
        if data_option == 'train':
            path = os.path.join(path, 'train_preprocessed_data.json')
            
        elif data_option == 'test':
            path = os.path.join(path, 'test_preprocessed_data.json')

        with open(path,'r') as file:
            data = json.load(file)
            
        self.data = data['data']
        
        self.emo_map = {
            'neutral': 0,
            'happy': 1,
            'surprise':2,
            'angry': 3,
            'sad':4,
            'disgust': 5,
            'fear': 6
            }
        
        # 감정 데이터 SoftMax로 변환
        for idx,data in enumerate(self.data):
            emo_list = [0]*7
            for emo in data['Emotion']:
                emo = self.emo_map[emo]
                emo_list[emo] += 1
            
            self.data[idx]['label'] = list(map((lambda x: x/sum(emo_list)), emo_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    def prepare_text_data(self,text_config):
        K = text_config.K
        for idx,data in enumerate(self.data):
            dialogue = data['utterance']#+data['history'][:K-1]
            #dialogue = '[SEP]'.join(dialogue)
            self.data[idx]['dialogue'] = dialogue

    def get_weight(self):
        weight = Counter([data['label'] for data in self.data])
        weight = [weight[i] for i in range(0,7)]
        sum_ = len(self.data)
        weight = [sum_/i for i in weight]
        return weight




if __name__ == '__main__':
    dataset = MERGEDataset(data_option='train',path='./data/')
    weight = dataset.get_weight()
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.sampler import WeightedRandomSampler

    labels = [data['label'] for data in dataset]
    counter = Counter(labels)
    counter = {k:len(labels)/v for k,v in counter.items()}
    import torch
    weight = [counter[i] for i in labels]
    sampler = WeightedRandomSampler(weight,len(weight))
    dataloader = DataLoader(dataset,batch_size=16,sampler=sampler,collate_fn= lambda x: (x,torch.Tensor([i['label']for i in x])))
    l = []
    for batch_x, batch_y in dataloader:
        l.extend(batch_y.tolist())
    print(Counter(l))
