import pandas as pd
import os
import json
import random
from sklearn.model_selection import train_test_split

PATH = './'
max_one_emotion = 4000
max_emotion = 6000
nclass = 7

json_path = os.path.join(PATH, 'data', 'total_data.json')
with open(json_path,'r') as file:
    base_json = json.load(file)

Middle_data = {"data": []}
Final_data = {"data" : []}
emo_count = {}


# 같은 값을 가진 감정 중, 전체 라벨 수가 적은 감정을 우선적으로 사용한다.
for idx, item in enumerate(base_json['data']):
    if not (os.path.isfile(os.path.join('/root/TOTAL',  item['wav']))):
        print('파일없음 삭제 :', os.path.join('/root/TOTAL',  item['wav']))
        del base_json['data'][idx]
        continue
    emo_dic = {}
    elem_list = []
    for emo in item['Emotion']:
        emo_dic[emo] = emo_dic.get(emo, 0) + 1
    for k, v in emo_dic.items():
        elem_list.append([v, k])
    elem_list.sort()
    top_emo = elem_list.pop()
    item['label'] = top_emo[1]
    if elem_list:
        if(top_emo[0] == elem_list[-1][0]):
            while(elem_list and top_emo[0] == elem_list[-1][0]):
                if (top_emo[1] in ['neutral', 'angry', 'sad']):
                    top_emo = elem_list.pop()
                else:
                    item['label'] = top_emo[1]
                    break
        Middle_data['data'].append(item)
    else:
        emo_count[top_emo[1]] = emo_count.get(top_emo[1], 0)
        if (emo_count[top_emo[1]] <= max_one_emotion): # 단독 감정을 가진 값이 너무 많이 포함되지 않게 한다.
            Final_data['data'].append(item)
            emo_count[top_emo[1]] += 1
        else:
            Middle_data['data'].append(item)
    
print("단독 감정 데이터 추출")
print(emo_count)

# Get Emotion Randomly
random.shuffle(Middle_data['data'])
while(Middle_data['data']):
    item = Middle_data['data'].pop()

    if (emo_count[item['label']] < max_emotion):
        Final_data['data'].append(item)
        emo_count[item['label']] += 1
    
    if (sum(list(emo_count.values())) >= max_emotion*nclass):
        print(max_emotion, nclass)
        break

print("각 감정당 최대" + str(max_emotion) + "개 추출")
print(emo_count)

print("데이터 저장")
# save preprocessed data
json_path = os.path.join(PATH, 'data', 'preprocessed_data.json')
with open(json_path,'w') as j:
    json.dump(Final_data,j,ensure_ascii=False, indent=4)


# Train Test Split
train_data, test_data = train_test_split(Final_data['data'], train_size=0.8, test_size=0.2, random_state=123, shuffle=True)

train_path = os.path.join(PATH, 'data', 'train_preprocessed_data.json')
print("Train 데이터셋 저장")
train_json = {'data' : train_data}
with open(train_path,'w') as j:
    json.dump(train_json,j,ensure_ascii=False, indent=4)

test_path = os.path.join(PATH, 'data', 'test_preprocessed_data.json')
print("Test 데이터셋 저장")
test_json = {'data' : test_data}
with open(test_path,'w') as j:
    json.dump(test_json,j,ensure_ascii=False, indent=4)

# Dateaset extract folder
import shutil
import os

PATH = './'
COPYPATH = os.path.join(PATH, 'TOTAL', 'Extracted_Dataset')
os.makedirs(COPYPATH, exist_ok=True)
for idx, item in enumerate(Final_data['data']):
    cur_path = os.path.join(PATH, item['wav'])
    destination = os.path.join(COPYPATH, item['wav'])
    try:
        shutil.copy(cur_path, destination)
    except:
        print(cur_path)
        del Final_data['data'][idx]