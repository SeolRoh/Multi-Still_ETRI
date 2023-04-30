import pandas as pd
import shutil
import json
import os

import numpy as np

def GenerateSubDir(PATH, COPYPATH):
    # 파일 복사 (현재 경로, 복사할 경로)
    def CopyFile(cur_path, COPYPATH):
        if not (os.path.isfile(COPYPATH)):
            shutil.copy(cur_path, COPYPATH)
        
    # 재귀적으로 파일 탐색(현재 경로, 복사할 경로, 파일의 태그(종류, wav, EDA ....))
    def SearchFiles(path, COPYPATH, tagname):
        for cur in os.listdir(path):
            cur_path = os.path.join(path, cur)
            if os.path.isdir(cur_path):
                SearchFiles(cur_path, COPYPATH, tagname)
            else:
                CopyFile(cur_path, os.path.join(COPYPATH, tagname+'_'+cur))

    os.makedirs(COPYPATH, exist_ok=True)
    for Ori in ["감정 분류를 위한 대화 음성 데이터셋"]:
        for cur in os.listdir(os.path.join(PATH, Ori)):
            cur_path = os.path.join(PATH, Ori, cur)
            if os.path.isdir(cur_path):
                SearchFiles(cur_path, COPYPATH, cur)
            else:
                CopyFile(cur_path, os.path.join(COPYPATH, cur))

def Setting_Emotion(series):
    encoder = {
        'neutral' : 'neutral',
        'angry' : 'angry',
        'sadness' : 'sad',
        'disgust' : 'disgust',
        'happiness' : 'happy',
        'fear' : 'fear',
        'surprise' : 'surprise'
    }
    data = series.values[3:]
    emotions = []

    for i in range(0, len(data), 2):
        emo = data[i].lower()
        emo = encoder[emo]
        count = data[i+1]+1
        emotions += [emo]*count
    return emotions
    
def Read_DataFrames(COPYPATH, file_name):
        df = pd.read_csv(os.path.join(COPYPATH, file_name), encoding="cp949")
        df = df[df.columns[:-2]]

        emotions = []
        wavid = []
        for i in range(len(df)):
            emotions.append(Setting_Emotion(df.iloc[i]))
            wavid.append(file_name[:-4]+ '_' + df['wav_id'].iloc[i]+".wav")
        df['Emotion'] = emotions
        df['wav'] = wavid
        df = df[['wav_id', 'wav', '발화문', 'Emotion']]
        df.columns = ['file_name', 'wav', 'utterance', 'Emotion']
        return df

os.chdir("/root/")
#if __name__ == '__main__':
# Move All files to one directory
PATH = './'
COPYPATH = os.path.join(PATH, "TOTAL")
GenerateSubDir(PATH, COPYPATH)


# 아까 모아놓았던 경로
base_df = pd.DataFrame(columns=['file_name', 'wav', 'utterance', 'Emotion'])

for file_name in ['4차년도.csv','5차년도.csv', '5차년도_2차.csv']:
    df = Read_DataFrames(COPYPATH, file_name)
    base_df = pd.concat([base_df, df], axis=0)

json_path = os.path.join(PATH, 'data', 'total_data.json')
with open(json_path,'r') as file:
    base_json = json.load(file)

for i in range(len(base_df)):
    item = {key:val for key, val in zip(base_df.columns, base_df.iloc[i].values)}
    base_json['data'].append(item)

# save preprocessed data
with open(json_path,'w') as j:
    json.dump(base_json,j,ensure_ascii=False, indent=4)




        
        