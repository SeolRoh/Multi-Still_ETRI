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

> 😊 Environment Setting
```bash
apt-get update && apt-get install -y
apt install ffmpeg -y
pip install numpy==1.22.3 pandas==1.4.2 scikit-learn transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
> 😊 Preprocessing
```bash
# 데이터 전처리
cd HappynJoy
# KEMDy19, KEMDy20 데이터 전처리
python KEMDy_my_preprocessing.py
# 외부데이터([AI Hub 감정 분류를 위한 대화 음성 데이터셋]) 전처리
#python external_data1_my_preprocessing.py
# 외부데이터([AI Hub 감정 분류용 데이터셋]) 전처리
#python external_data2_my_preprocessing.py
# 감정의 불균형을 막기 위해, 하나의 특정 감정을 일정값 이하로 정리
python data_balancing.py
```

+ 7가지 감정 레이블의 데이터 불균형 완화 전후 분포 비교

![](https://github.com/jo1132/HappynJoy/blob/main/images/%ED%95%99%EC%8A%B5%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%9D%BC%EB%B2%A8%EB%B6%84%ED%8F%AC_%EB%8F%99%EA%B7%B8%EB%9D%BC%EB%AF%B8_0428.png)


> 😊 Train
```bash
# 멀티모달 크로스 어텐션 교사 모델 학습
python my_train_crossattention.py --model_name multimodal_teacher
# 교사모델을 활용해, 데이터셋에 증류 데이터 추가
python Distill_knowledge.py --model_name multimodal_teacher_epoch29
# 멀티모달 학생 모델 지식증류 훈련
python mini_my_train_crossattention.py --model_name multimodal_student
# 문자모달 학생 모델 지식증류 훈련
python mini_my_train_crossattention.py --text_only True --model_name text_student
# 음성모달 학생 모델 지식증류 훈련
python mini_my_train_crossattention.py --audio_only True --model_name audio_student
```

> 😊 Test
```bash
# pt file will generate  in ckpt after train
```
> 😊 setup.sh
```bash
# 업데이트
apt-get update && apt-get upgrade -y
# 음성 데이터를 처리하기 위한 모듈 설치
apt install ffmpeg -y
# 학습을 위한 파이썬 라이브러리 설치
pip install numpy==1.22.3 pandas==1.4.2 scikit-learn transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1 moviepy
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```


> 😁 Directory
- 코드 구현을 위해서는 ETRI에서 제공하는 파일(KEMDy19 & KEMDy20)과 AI Hub 감정 데이터 파일이 알맞은 위치에 있어야합니다.
```
+--KEMDy20
      +--annotation
      +--wav
      # train과 inference 속도를 향상시키기 위해 pretrained Wav2Vec2모델에서 연산한 결과를 미리 저장하여 활용하였음.
            +--audio_embeddings    
                  +--hidden_state.json    
                  +--extract_feature.json
      # train과 inference 속도를 향상시키기 위해 pretrained Wav2Vec2모델에서 연산한 결과를 미리 저장하여 활용하였음.
            +--hidden_states
                +-- {file_name}.pt
      # AI Hub 감성대화 말뭉치 file들이 저장된 폴더
            +--emotiondialogue
                +--F_000001.wav
                ...
                +--M_005000.wav
            +--Sessoion01
            ...
            +--Session40
      +--TEMP
      +--IBI
      +--EDA
+--data
      +--processed_KEMDy20.json   # KEMDy20데이터와 감성대화 말뭉치를 전처리한 파일
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
| Text Encoder | pretrained Electra | beomi/KcELECTRA-base | 

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