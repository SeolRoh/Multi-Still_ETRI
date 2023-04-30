# Multi-Still: A lightweight Multi-modal Cross Attention Knowledge Distillation method for real-time Emotion Recognition

## 제2회 ETRI 휴먼이해 인공지능 논문경진대회(2023)
### 본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다


> 😊 대회 소개
+ 인간과 교감할 수 있는 인공지능을 구현하기 위한 목적으로 개최 되었습니다.
+ 사람의 행동과 감정을 이해하는 기술 연구를 가능토록 하기위해 구축한 데이터셋을 활용하여 휴먼이해 인공지능 기술 연구를 확산시키고자 합니다.
+ 이에 창의적은 연구를 발굴하고자 합니다.

> 😊 주최/주관
+ 주최 : 한국전자통신연구원 (ETRI)
+ 후원 : 과학정보기술통신부, 국가과학기술연구회 (NST)
+ 운영 : 인공지능팩토리 (AIFactory)

> 😊 논문 주제
+ 멀티모달 감정 데이터셋 활용 감정 인식 기술 분야
+ 논문주제:  Emotion Recognition in Conversation (ERC)분야
+ Multi-Still: 실시간 감정 인식을 위한 경량화된 멀티모달 교차 어텐션 지식 증류 방법

      
      * Emotion Recognition in Conversation이란?
      두 명 이상의 참여자 간의 대화(dialogue)과정에서 대화 참여자의 감정을 인식 또는 예측하기 위한 감정인식 연구분야입니다.
      

> 😊 활용 데이터: ETRI 한국어 감정 데이터셋 활용 연구

- 📁  [KEMDy19 (성우 대상 상황극) 데이터셋](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)
- 📁  [KEMDy20 (일반인 대상 자유발화) 데이터셋](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)


> 😊 Environment
```
python version : python3.9
OS type : WSL
requires packages: {
      'numpy==1.22.3',
      'pandas==1.4.2',
      'torch==1.11.0+cu113',
      'torchaudio==0.11.0+cu113',
      'scikit-learn',
      'transformers==4.18.0',
      'tokenizers==0.12.1',
      'soundfile==0.10.3.post1'
}
```

> 😊 Docker container run
```bash
docker container run -d -it --name multi_still --gpus all python:3.9
```

> 😊 Environment Setting
```bash
git clone https://github.com/SeolRoh/Multi-Still_ETRI.git
cd Multi-Still_ETRI
bash setup.sh
```
> 😊 Preprocessing
```bash
# 데이터 전처리
bash Data_Preprocessing.sh
```

+ 7가지 감정 레이블의 데이터 불균형 완화 전후 분포 비교

![](https://github.com/jo1132/HappynJoy/blob/main/images/%ED%95%99%EC%8A%B5%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%9D%BC%EB%B2%A8%EB%B6%84%ED%8F%AC_%EB%8F%99%EA%B7%B8%EB%9D%BC%EB%AF%B8_0428.png)


> 😊 Train
```bash
# 교사모델을 활용해, 데이터셋에 증류 데이터(Softmax) 추가
#--teacher_name 옵션으로 MultiModal 교사모델의 이름_epoch수를 입력한다.
#--data_path 옵션으로 softmax 데이터를 추가할 기존 데이터셋의 경로를 입력 (기본값, "data/train_preprocessed_data.json")
python Distill_knowledge.py --teacher_name multimodal_teacher_epoch4 

# miniconfig.py 를 수정해서 Epoch를 포함한 하이퍼파라미터 변경
# 멀티모달 학생 모델 지식증류 훈련
python KD_my_train_crossattention.py --model_name multimodal_student
# 문자모달 학생 모델 지식증류 훈련
python KD_my_train_crossattention.py --model_name text_student --text_only True 
# 음성모달 학생 모델 지식증류 훈련
python KD_my_train_crossattention.py --model_name audio_student --audio_only True
```

> 😊 Test
```bash
# pt 파일은 훈련의 5번째 Epoch마다 생성됨. (예: 5, 10, 11....)
# 여러 파일을 테스트 하기위해 test_all파일에 복사
cp ckpt/* ckpt/test_all/
python my_test.py --all
```


> 😁 Directory
- 코드 구현을 위해서는 ETRI에서 제공하는 파일(KEMDy19 & KEMDy20)과 AI Hub 감정 데이터 파일이 알맞은 위치에 있어야합니다.
```
+--Multi-Still_ETRI
      +--KEMDy19
            +--annotation
            +--ECG
            +--EDA
            +--TEMP
            +--wav
            # train과 inference 속도를 향상시키기 위해 미리 훈련된 Wav2Vec2모델에서 인코딩한 결과를 미리 저장하여 활용.
      +--KEMDy20
            +--annotation
            +--wav
            # train과 inference 속도를 향상시키기 위해 pretrained Wav2Vec2모델에서 연산한 결과를 미리 저장하여 활용하였음.
            +--TEMP
            +--IBI
            +--EDA
      +--감정 분류를 위한 대화 음성 데이터셋 (선택)
            # 음성 데이터가 포함되어있는 폴더
            +--4차년도
            +--5차년도
            +--5차년도_2차
            # 각 음성데이터에 대한 정보가 담겨있는 csv파일
            +--4차년도.csv
            +--5차년도.csv
            +--5차년도_2차.csv
      +--감정분류용 데이터셋 (선택)
            # 영상 및 이미지가 포함되어 있는 폴더
            +--0~9_감정분류_데이터셋
            +--10~19_감정분류_데이터셋
            +--20~29_감정분류_데이터셋
            +--30~39_감정분류_데이터셋
            +--40~49_감정분류_데이터셋
            +--50~59_감정분류_데이터셋
            +--60~69_감정분류_데이터셋
            +--70~79_감정분류_데이터셋
            +--80~89_감정분류_데이터셋
            +--90~99_감정분류_데이터셋
            # 각 영상 및 이미지정보의 스크립트 데이터
            +--Script.hwd 
            # 각 영상 및 이미지정보의 참가자 정보 데이터
            +--참가자정보.xlsx
            
      +--data
            +--total_data.json   # 모든 데이터셋을 전처리한 파일
            +--preprocessed_data.json   # 모든 데이터셋에서 감정 분포를 완화한 파일
            +--test_preprocessed_data.json   # preprocessed_data.json에서 test데이터를 추출한 파일
            +--train_preprocessed_data.json   # preprocessed_data.json에서 train데이터를 추출한 파일
      +--models
            +--module_for clossattention
            +--multimodal.py
            +--multimodal_attention
            +--multimodal_cross_attention
            +--multimodal_mixer      
      +--merdataset.py
      +--preprocessing.py
      +--utils.py
      +--test.py
      +--config.py
      +--train.py
      +--train_crossattention.py
      +--train_mixer.py
      ```

> 😆 Base Model
| Encoder | Architecture | pretrained-weights | 
| ------------ | ------------- | ------------- |
| Audio Encoder | pretrained Wav2Vec 2.0 | kresnik/wav2vec2-large-xlsr-korean |
| Text Encoder | pretrained Electra | monologg/koelectra-base | 

> 😃 Arguments
- train.py
- train_crossattention.py
- train_knowledge_distillation.py
- test.py

> 😀 Model Architecture
- `Multi-Still` 경량화 기술 중 하나인 지식 증류 (Knowledge Distillation)를 사용하여 실시간 감정인식을 위한 멀티모달 구조를 경량화하는 방법 
- 👩‍🏫➡👨‍💻 Muti-Still Architecture
![](https://velog.velcdn.com/images/dkddkkd55/post/21aa86c8-fa0e-4669-955e-d6f113547a9b/image.png)
- 👩‍🏫 Teacher Model
![](https://velog.velcdn.com/images/dkddkkd55/post/a6ca8342-0faa-4990-a334-3694b12a2f07/image.png)

> 😆 Experiments 
+ 텍스트 모델(KoELECTRA)
+ 오디오 모델(WAV2VEC 2.0)
+ 교사 모델(Multimodal Cross-Attention)
+ 학생모델((a)Text-OnlyStudent, (b)Audio-OnlyStudent, (c)MultimodalStudent)

![](https://github.com/jo1132/HappynJoy/blob/main/images/Experiments.png)


> 🙂 References

+ [1]  Xu, Peng, Xiatian Zhu, and David A. Clifton. "Multimodal learning with transformers: A survey." arXiv preprint arXiv:2206.06488 (2022).

+ [2] Tao, Jiang, Zhen Gao, and Zhaohui Guo. "Training Vision Transformers in Federated Learning with Limited Edge-Device Resources." Electronics 11.17, 2638. (2022).

+ [3] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. ”Distilling the knowledge in a neural network.” arXiv preprint arXiv:1503.02531 2.7 (2015).

+ [4] Gou, J., Yu, B., Maybank, S. J., & Tao, D. “Knowledge distillation: A survey”. International Journal of Computer Vision, 129, 1789-1819. (2021).

+ [5] K. J. Noh and H. Jeong, “KEMDy19,” https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR.

+ [6] Noh, K.J.; Jeong, C.Y.; Lim, J.; Chung, S.; Kim, G.; Lim, J.M.; Jeong, H. Multi-Path and Group-Loss-Based Network for Speech Emotion Recognition in Multi-Domain Datasets. Sensors 2021, 21, 1579. https://doi.org/10.3390/s21051579. 

+ [7] K. J. Noh and H. Jeong, “KEMDy20,” https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR.

+ [8] Thabtah, Fadi, et al. "Data imbalance in classification: Experimental evaluation." Information Sciences 513 : 429-441 (2020).

+ [9] Park Jangwon, “KoELECTRA: Pretrained ELECTRA Model for Korean”https://github.com/monologg/KoELECT RA (2020).

+ [10] Baevski, Alexei, et al. “wav2vec 2.0: A framework for self-supervised learning of speech representations”. Advances in Neural Information Processing Systems, 33, 12449-12460, (2020).


> 🙂 Contact
+ Hyun-Ki Jo : jhk1132@khu.ac.kr
+ Yu-Ri Seo : yuri0329@khu.ac.kr
+ Seol Roh : seven800@khu.ac.kr