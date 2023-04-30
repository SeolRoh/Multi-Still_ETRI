import pandas as pd
import shutil
import json
import os

import numpy as np
import moviepy.editor as mp

def GenerateSubDir(PATH, COPYPATH):
    # 파일 복사 (현재 경로, 복사할 경로)
    def CopyFile(cur_path, COPYPATH):
        if not (os.path.isfile(COPYPATH)):
            file_name, ext = os.path.splitext(cur_path)
            if(ext == '.csv'):
                shutil.copy(cur_path, COPYPATH)
            elif(ext == ".wav"):
                audio = mp.AudioFileClip(cur_path)
                audio.write_audiofile(COPYPATH)

        
    # 재귀적으로 파일 탐색(현재 경로, 복사할 경로, 파일의 태그(종류, wav, EDA ....))
    def SearchFiles(path, COPYPATH, tagname):
        for cur in os.listdir(path):
            cur_path = os.path.join(path, cur)
            if os.path.isdir(cur_path):
                SearchFiles(cur_path, COPYPATH, tagname)
            else:
                CopyFile(cur_path, os.path.join(COPYPATH, tagname+'_'+cur))

    os.makedirs(COPYPATH, exist_ok=True)
    for Ori in ["감정분류용 데이터셋"]:
        for cur in os.listdir(os.path.join(PATH, Ori)):
            cur_path = os.path.join(PATH, Ori, cur)
            if os.path.isdir(cur_path):
                SearchFiles(cur_path, COPYPATH, "감정분류용")
            else:
                CopyFile(cur_path, os.path.join(COPYPATH, cur))
    


def Read_DataFrames(COPYPATH, file_name):
    script_df = pd.read_csv(os.path.join(COPYPATH, file_name), encoding="cp949")
    script_df = script_df[script_df.columns[:3]]

    encoder = {
            'neutral' : 'neutral',
            'angry' : 'angry',
            'sadness' : 'sad',
            'sad' : 'sad',
            'disgust' : 'disgust',
            'happiness' : 'happy',
            'fear' : 'fear',
            'surprise' : 'surprise'
    }

    script_dict = {}
    for i in range(len(script_df)):
        number = int(script_df['number'].iloc[i])
        script = script_df['script'].iloc[i]
        emotion = [script_df['emotion'].iloc[i]]
        emotion = [encoder[emo.lower()] for emo in emotion]
        script_dict[number] = {
            'script' : script,
            'emotion' : emotion
        }
    
    file_names = []
    for file in os.listdir(COPYPATH):
        file_name, ext = os.path.splitext(file)
        if(file_name[:5] == '감정분류용' and ext == '.wav'):
            file_names.append(file)
    
    data = []
    for item in file_names:
         name, ext = os.path.splitext(item)
         script_key = int(name[-3:])
         if script_dict.get(script_key, 0):
             data.append({
                 'wav' : item,
                 'file_name' : name,
                 'utterance' : script_dict[script_key]['script'],
                 'Emotion' : script_dict[script_key]['emotion']
             })
    return data

os.chdir("/root/")
#if __name__ == '__main__':
# Move All files to one directory
PATH = './'
COPYPATH = os.path.join(PATH, "TOTAL")
GenerateSubDir(PATH, COPYPATH)

data = Read_DataFrames(COPYPATH, '스크립트.csv')

json_path = os.path.join(PATH, 'data', 'total_data.json')
with open(json_path,'r') as file:
    base_json = json.load(file)

base_json['data'].extend(data)


# save preprocessed data
with open(json_path,'w') as j:
    json.dump(base_json,j,ensure_ascii=False, indent=4)