import pandas as pd
import os
import json
import random
from sklearn.model_selection import train_test_split

PATH = './'
max_one_emotion = 4000
max_emotion = 6000
nclass = 7

# 파일 읽기
json_path = os.path.join(PATH, 'data', 'total_data.json')
with open(json_path,'r') as file:
    base_json = json.load(file)

# 감정 클래스 분포 확인
Final_data = {"data" : []}
emo_count = {}
for idx, item in enumerate(base_json['data']):
    if not (os.path.isfile(os.path.join(PATH, 'TOTAL',  item['wav']))):
        print('파일없음 삭제 :', os.path.join(PATH, 'TOTAL',  item['wav']))
        del base_json['data'][idx]
        continue

    for emo in item['Emotion']:
        emo_count[emo] = emo_count.get(emo, 0) + 1
    
print("감정 클래스 분포 확인")
print(emo_count)

# 감정 데이터 셔플
random.shuffle(base_json['data'])

print("데이터 저장")
# save preprocessed data
json_path = os.path.join(PATH, 'data', 'preprocessed_data.json')
with open(json_path,'w') as j:
    json.dump(base_json,j,ensure_ascii=False, indent=4)


# 훈련, 테스트 데이터 split
train_data, test_data = train_test_split(base_json['data'], train_size=0.8, test_size=0.2, random_state=123, shuffle=True)

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
