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
    
    #"KEMDy19", "KEMDy20" 파일 탐색
    os.makedirs(COPYPATH, exist_ok=True)
    for Ori in ["KEMDy19", "KEMDy20"]:
        for cur in os.listdir(os.path.join(PATH, Ori)):
            cur_path = os.path.join(PATH, Ori, cur)
            SearchFiles(cur_path, COPYPATH, cur)

#데이터셋이 가지고 있는 감정 데이터 DataFrame으로  
def Read_DataFrames(COPYPATH, head, tail, n_sess, cols):
        base_df = pd.DataFrame(columns=['Segment ID', 'Emotion', 'Valence', 'Arousal']) 

        for i in range(1, n_sess+1):
            file_path = os.path.join(COPYPATH, head+"{0:02d}".format(i)+tail)

            if os.path.isfile(file_path):
                df = cut_df(pd.read_csv(file_path), cols)
                base_df = pd.concat([base_df, df], axis=0)

        return base_df

def ndarrayToList(x):
    return x[0].tolist()

# DataFrame 세팅
def cut_df(df, filter_cols):
    df = df.iloc[1:].copy()
    columns = []
    for item in df.columns:
        if item[:4] == "Eval":
            columns.append(item)
    temp = df[columns]
    Emotion = []
    for i in range(len(temp)):
        Emotion.append([temp[columns].iloc[i].values])
    df['Total Evaluation'] = list(map(ndarrayToList, Emotion))

    df['index'] = df['Segment ID']
    df = df.set_index(['index'])
    df = df[filter_cols]
    df.columns = ['Segment ID', 'Emotion', 'Valence', 'Arousal']
    return df

# KEMDy19 데이터셋 M, F 합치기
def merge_data(F_df, M_df, neutral=True):
    aro = 0
    vals = 0
    base_df = pd.DataFrame(columns=['Segment ID', 'Emotion', 'Valence', 'Arousal'])

    for i in range(len(F_df)):
        F = F_df.iloc[i]
        M = M_df.iloc[i]

        if F['Segment ID'] != M['Segment ID']:
            print('ID diffrent Error', F['Segment ID'], M['Segment ID'])

        else:
            ID = F['Segment ID']
            vals = (float(F['Valence']) + float(M['Valence'])) / 2
            aro = (float(F['Arousal']) + float(M['Arousal'])) / 2
    
            emo = F['Emotion'] + M['Emotion']
            df = pd.DataFrame([ID, emo, vals, aro], index=['Segment ID', 'Emotion', 'Valence', 'Arousal']).T
            base_df = pd.concat([base_df, df], axis=0)

    return base_df

# 음성파일에 맞는 스크립트 데이터 불러오기
def Read_txt(path):
    script = ''
    if os.path.isfile(path):
        try:
            with open(path, 'rt', encoding='CP949') as file:
                script = file.read()
        except:
            with open(path, 'rt', encoding='UTF-8') as file:
                script = file.read()
    else:
        print('No file', path)
    return script

def GetWavPath(x):
    return 'wav_'+x+'.wav'

if __name__ == '__main__':
    # Move All files to one directory
    PATH = './'
    COPYPATH = os.path.join(PATH, "TOTAL")
    GenerateSubDir(PATH, COPYPATH)
    
    # 저장할 데이터 columns
    df = pd.DataFrame(columns=['Segment ID', 'Emotion', 'Valence', 'Arousal'])

    # KEMDy20 에서 사용할 columns
    cols1 = ['Segment ID', 'Total Evaluation', ' .1', ' .2']
    cols2 = cols1.copy()
    # KEMDy19 에서 사용할 columns
    cols2[2] = 'Unnamed: 11'
    cols2[3] = 'Unnamed: 12'

    df1 = Read_DataFrames(COPYPATH, "annotation_Sess", "_eval.csv", 40, cols1)
    df2 = Read_DataFrames(COPYPATH, "annotation_Session", "_F_res.csv", 20, cols2)
    df3 = Read_DataFrames(COPYPATH, "annotation_Session", "_M_res.csv", 20, cols2)
    
    # KEMDy19, KEMDy20 합치기
    df = pd.concat([df, merge_data(df2, df3)], axis=0)
    df = pd.concat([df, df1], axis=0)

    df = df.sort_values(by=['Segment ID'])
    df = df.reset_index().drop(labels=['index'], axis=1)
    df['Script'] = [0]*len(df)

    # 음성 데이터 스크립트 불러오기
    for idx in range(len(df)):
        if type(df.iloc[idx]['Segment ID']) == str:
            SegID = "wav_"+df.iloc[idx]['Segment ID']
            file_name = SegID+'.txt'
            df['Script'].iloc[idx] = Read_txt(os.path.join(COPYPATH, file_name))
        else:
            print(df.iloc[idx]['Segment ID'])
            df = df.drop(idx, axis=0)

    # 음성데이터 이름 저장
    df['Audio'] = df['Segment ID'].apply(GetWavPath)
    df = df.dropna(axis=0)
    df = df[['Segment ID', 'Audio', 'Script', 'Emotion']]
    #df.to_csv(os.path.join(PATH, 'merged_data.csv'), encoding="utf-8-sig", index=False)

    # 데이터 
    base_json = {"data":[]}
    for i in range(len(df)):
        data = {
            "file_name" : df.iloc[i]['Segment ID'],
            "wav" : df.iloc[i]['Audio'],
            "utterance" : df.iloc[i]['Script'],
            "Emotion" : df.iloc[i]['Emotion']
        }
        base_json["data"].append(data)
    '''
    for sess in base_json.keys():
        for script in base_json[sess].keys():
            for i in range(len(base_json[sess][script])):
                base_json[sess][script][i]['history'] = [dic['utterance'] for dic in base_json[sess][script][:i]][::-1]
    '''
    
    # save preprocessed data
    os.makedirs(os.path.join(PATH, 'data'), exist_ok=True)
    with open(os.path.join(PATH, 'data', 'total_data.json'),'w') as j:
        json.dump(base_json,j,ensure_ascii=False, indent=4)




        
        
