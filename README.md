- [Dialogue Summarization | 일상 대화 요약](#dialogue-summarization--일상-대화-요약)
    - [Team](#team)
    - [0. Overview](#0-overview)
        - [Environments](#environment)
        - [Requirements](#requirements)
    - [1. Competiton Info](#1-competiton-info)
        - [Introduce](#introduce)
        - [Timeline](#timeline)
    - [2. Components](#2-components)
        - [Directory](#directory)
    - [3. Data descrption](#3-data-descrption)
        - [Dataset overview](#dataset-overview)
        - [EDA & Data Processing](#eda--data-processing)
            - [1. 오타 수정](#1-오타-수정)
            - [2. 제대로 작성되지 않은 Topic](#2-제대로-작성되지-않은-topic)
            - [3. [ ]로 감싸진 대화문 전체 제거](#3---로-감싸진-대화문-전체-제거)
            - [4. Data performance](#4-data-performance)
    - [4. Modeling](#4-modeling)
        - [Model descrition](#model-descrition)
            - [김주형](#김주형)
                - [Dialogues 데이터 수치형 변환](#dialogues-데이터-수치형-변환)
                - [Dialogue 데이터 클러스터링](#dialogue-데이터-클러스터링)
            - [성명기](#성명기)
                - [가설1. 입출력의 길이를 늘리면 성능이 좋아질 것이다.](#가설1-입출력의-길이를-늘리면-성능이-좋아질-것이다)
                - [가설2. 학습률을 늘리면 성능이 좋아질 것이다.](#가설2-학습률을-늘리면-성능이-좋아질-것이다)
                - [가설3. 학습률 스케줄러 유형을 변경하면 성능이 좋아질 것이다.](#가설3-학습률-스케줄러-유형을-변경하면-성능이-좋아질-것이다)
                - [가설4. T5 모델중 Large모델링을 하면 성능이 대폭 상승할 것이다.](#가설4-t5-모델중-large모델링을-하면-성능이-대폭-상승할-것이다)
            - [임동건](#임동건)
                - [BART계열 모델 테스트](#bart계열-모델-테스트)
                - [num_beam 파라미터 수정](#num_beam-파라미터-수정)
                - [learning_rate 파라미터 수정](#learning_rate-파라미터-수정)
            - [유정수](#유정수)
            - [장재성](#장재성)
    - [5. Result](#5-result)
        - [Leader Board](#leader-board)
        - [Presentation](#presentation)

<br>

---
<br>

# Dialogue Summarization | 일상 대화 요약
## Team

| ![성명기](https://avatars.githubusercontent.com/u/104310191?v=4) | ![김주형](https://avatars.githubusercontent.com/u/95218618?v=4) | ![임동건](https://avatars.githubusercontent.com/u/125024589?v=4) | ![유정수](https://avatars.githubusercontent.com/u/50096716?v=4) | ![장재성](https://avatars.githubusercontent.com/u/165862584?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| 팀장, EDA&전처리, T5 모델링 |   BERT기반 토크나이즈, 클러스터링     |  전처리, 모델링     |   Few-Shot learning | 데이터 기계번역 모델링  |
|            [성명기](https://github.com/SUNGMYEONGGI)             |            [김주형](https://github.com/AjouKim)             |            [임동건](https://github.com/LimDG1981)            |            [유정수](https://github.com/Dream-Forge-Studios)             |            [장재성](https://github.com/mirrbandi)             |

## 0. Overview
### Environment
-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements    
```bash
pip install -r requirements.txt
```

## 1. Competiton Info

### Introduce
Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 

일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.

이를 돕기 위해, 우리는 이번 대회에서 일상 대화를 바탕으로 요약문을 생성하는 모델을 구축합니다!

![image](https://github.com/HojunJ/conventional-repo/assets/76687996/1ba682aa-f341-4e84-a788-57994fa845ba)

참가자들은 대회에서 제공된 데이터셋을 기반으로 모델을 학습하고, 대화의 요약문을 생성하는데 중점을 둡니다. 이를 위해 다양한 구조의 자연어 모델을 구축할 수 있습니다.

제공되는 데이터셋은 오직 "대화문과 요약문"입니다. 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있습니다.

참가자들은 이러한 비정형 텍스트 데이터를 고려하여 모델을 훈련하고, 요약문의 생성 성능을 높이기 위한 최적의 방법을 찾아야 합니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

> input : 249개의 대화문  
> output : 249개의 대화 요약문

### Timeline

- August 29, 2024 ~ September 10, 2024

## 2. Components

### Directory

e.g.
```
├── JANG JAESEONG
│   └── CODE
├── KIM JUHYUNG
│   └── CODE
├── LIM DONGGUN
│   └── CODE
├── SEONG MYEONGGI
│   └── CODE
├── YU JEONGSU
│   └── CODE
└── README.md
```

## 3. Data descrption

### Dataset overview
- train.csv & dev.csv
    - Column Name : fname, dialouge, summary, topic
    - fname : Data Index
    - dialouge : 2~7인 대화문
    - summary : dialouge 요약문
    - topic : 대화문의 주제

- test.csv
    - Column Name : fname, dialouge
    - fname : Data Index
    - dialouge : 2~7인 대화문

### EDA & Data Processing
#### 1. 오타 수정
<details>
<summary>train_5385</summary>

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%8B%E1%85%A9%E1%84%90%E1%85%A1%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC(train_5385).png?raw=true)

#Person2#: 먼저, 이것은 19세기 초 배경ㅇ로 설정된 로맨스 소설이에요.
→ 먼저, 이것은 19세기 초 배경으로 설정된 로맨스 소설이에요.
</details>

<details>
<summary>train_7201</summary>

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%8B%E1%85%A9%E1%84%90%E1%85%A1%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC(train_7201).png?raw=true)

#Person1#: 이제 그만. 너는 아직ㅍ알맞는 사람을 만나지 못했을 뿐이고,
→ 이제 그만. 너는 아직 알맞는 사람을 만나지 못했을 뿐이고,
</details>

<details>
<summary>train_9677</summary>

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%8B%E1%85%A9%E1%84%90%E1%85%A1%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC(train_9677).png?raw=true)

#Person1#: 이제 그만. 너는 아직ㅍ알맞는 사람을 만나지 못했을 뿐이고, 
→ 이제 그만. 너는 아직 알맞는 사람을 만나지 못했을 뿐이고,
</details>

<details>
<summary>train_12181</summary>

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%8B%E1%85%A9%E1%84%90%E1%85%A1%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC(train_12181).png?raw=true)

#Person1#:카페 근처에서. 나는 CD 가게에 들어가서 CD를 보는 척했ㄷ거든.
→ 카페 근처에서. 나는 CD 가게에 들어가서 CD를 보는 척했거든. 
</details>

#### 2. 제대로 작성되지 않은 Topic
일부 데이터의 Topic이 제대로 작성되지 않은 경우가 있었다. 이를 수정하기 위해 대화문을 통해 Topic을 추론하여 수정하였다.

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%90%E1%85%A9%E1%84%91%E1%85%B5%E1%86%A8%2024,26%E1%84%80%E1%85%A2%20%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5.png?raw=true)
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%90%E1%85%A9%E1%84%91%E1%85%B5%E1%86%A8%E1%84%8B%E1%85%B5%E1%84%89%E1%85%B2%202%E1%84%80%E1%85%A2.png?raw=true)

```python
# train_3804에 topic을 '미국의 학교'로 변경
# train_7595에 topic을 '12살인 동생이 있다'로 변경

train_df['topic'][3804] = '미국의 학교'
train_df['topic'][7595] = '12살인 동생이 있다'
```

#### 3. [  ]로 감싸진 대화문 전체 제거
제거하는 이유 : [ * ]로 감싸진 대화문을 제거해도 대화에 전혀 이질감이 느껴지지 않고 자연스러움
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%83%E1%85%A2%E1%84%80%E1%85%AA%E1%86%AF%E1%84%92%E1%85%A9%20%E1%84%8C%E1%85%A6%E1%84%80%E1%85%A5%20%E1%84%8B%E1%85%A8%E1%84%89%E1%85%B5%20%E1%84%83%E1%85%A2%E1%84%92%E1%85%AA%E1%84%86%E1%85%AE%E1%86%AB.png?raw=true)

#### 4. Data performance
오타만 수정한 데이터와 오타 + [ ] 제거한 데이터를 성능 비교했을 때 [ ]로 감싸진 대화문을 제거한 데이터가 성능이 좋았다.
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%A1%E1%84%80%E1%85%A9%E1%86%BC%20%E1%84%92%E1%85%AE%E1%86%AB%E1%84%85%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AE%20wandb%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA.png?raw=true)

## 4. Modeling

### Model descrition

#### 김주형
##### *Dialogues 데이터 수치형 변환*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/Dialogues%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%AE%E1%84%8E%E1%85%B5%E1%84%92%E1%85%A7%E1%86%BC%20%E1%84%87%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB.png?raw=true)
- 대화 내용들의 분류가 있음을 확인하고 토픽 분포 등을 분석함
- Dialogue를 특징에 따라 나누기 위해 클러스터링 진행
- 클러스터링을 위해 Bert 모델로 Dialogue 데이터를 임베딩하여 수치로 이루어진 데이터로 변환

##### *Dialogue 데이터 클러스터링*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/Dialogue%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8F%E1%85%B3%E1%86%AF%E1%84%85%E1%85%A5%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%E1%84%85%E1%85%B5%E1%86%BC.png?raw=true)
- Dialogue를 특징에 따라 나누기 위해 클러스터링 진행
- K-means 클러스터링 결과를 PCA로 시각화 하였고, 5개의 군집으로 데이터를 구분할 수 있다는 것을 알 수있음
- 군집 데이터 별로 CSV 파일을 만들어 진행.

#### 성명기
##### *가설1. 입출력의 길이를 늘리면 성능이 좋아질 것이다.*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/length%20%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%92%E1%85%AE%20wandb%20%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA.png?raw=true)
- decoder_max_len과 encoder_max_len 사이즈를 각각 200과 1024로 늘림
    - 길이를 늘리면 CUDA Memory 이슈가 발생해서 Device Batch Size를 줄여줘야함
- generation_max_length를 200 or 50으로 조정
    - 50으로 조정시 기본값이었던 100보다 성능이 저하
    - 200으로 조정시 기본값이었던 100보다 성능이 상승
- generation_max_length의 값이 길수록 Rouge의 값이 좋아지지만 처리속도가 느려짐
- 손실 값은 대부분의 설정에서 학습이 진행됨에 따라 감소하며 손실 값의 차이는 거의 비슷

##### *가설2. 학습률을 늘리면 성능이 좋아질 것이다.*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/learning_rate%20%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%92%E1%85%AE%20wandb%20%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA.png?raw=true)
- ROUGE 점수는 lr 5e-05가 초반에는 좋지만, 훈련이 진행됨에 따라 오히려 성능이 하락하는 경향을 보이며, lr 1e-05는 꾸준히 성능이 증가
- 손실의 경우, lr 5e-05 설정은 3k 단계 이후 손실이 다시 증가하는 것을 볼 수 있음. 반면, lr 3e-05와 lr 1e-05는 지속적으로 감소
- 실행 시간은 학습률이 낮을수록 더 안정적으로 감소하는 경향을 보임

##### *가설3. 학습률 스케줄러 유형을 변경하면 성능이 좋아질 것이다.*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/lr_type%20%E1%84%89%E1%85%AE%E1%84%8C%E1%85%A5%E1%86%BC%20wandb%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA.png?raw=true)
- lr_scheduler_type에 cosine, linear, cosine_with_restarts 3가지 실험
- ROUGE-1 점수가 훈련 단계에 따라 증가하며 초반에 성능이 급격히 상승하다가 후반에서 점차 안정화
- cos_with_restarts가 다른 설정보다 약간 높은 성능을 보임
- 평가 손실은 대부분의 설정에서 학습이 진행됨에 따라 감소하며 손실 값의 차이는 거의 비슷

##### *가설4. T5 모델중 Large모델링을 하면 성능이 대폭 상승할 것이다.*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/T5%20%E1%84%89%E1%85%A5%E1%86%BC%E1%84%82%E1%85%B3%E1%86%BC%E1%84%8C%E1%85%B5%E1%84%91%E1%85%AD.png?raw=true)

#### 임동건
##### *BART계열 모델 테스트*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/(%E1%84%83%E1%85%A9%E1%86%BC%E1%84%80%E1%85%A5%E1%86%AB)bart%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF.png?raw=true)

##### *num_beam 파라미터 수정*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/(%E1%84%83%E1%85%A9%E1%86%BC%E1%84%80%E1%85%A5%E1%86%AB)num_beam.png?raw=true)
- 빔 서치는 모델이 단어(토큰)를 하나씩 생성할 때, 단순히 매 스텝마다 가장 가능성이 높은 단어만 선택하는 것이 아니라, 여러 가지 후보 경로를 동시에 유지하며 가장 가능성이 높은 시퀀스를 찾는 방법
- beam search 값을 높여서 더 높은 품질의 요약을 생성

##### *learning_rate 파라미터 수정*
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/(%E1%84%83%E1%85%A9%E1%86%BC%E1%84%80%E1%85%A5%E1%86%AB)lr.png?raw=true)
- leaning_rate를 1e-6, 2.5e-6, 5e-6, 1e-5 수정 후 wandb를 활용한 성능 확인
- 값이 낮을수록 속도는 느리지만 지속적인 학습 

#### 유정수
*few shot learning이란?*
> LLM에게 몇 가지 예시를 제공하여  학습하는 방법입니다.
새로운 작업에 빠르게 적응해야 하거나 학습 데이터가 부족한 상황에서 특히 유용합니다. 또한, LLM을 특정 분야나 스타일에 맞춰 미세 조정하는 데에도 활용될 수 있습니다.

*실험내용*

- 목표: test 데이터와 가장 유사한 train 데이터를 검색 후 few shot learning을 통한 성능 비교
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%83%E1%85%A1%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8B%E1%85%A5%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%89%E1%85%A5%20%E1%84%80%E1%85%A5%E1%86%B7%E1%84%89%E1%85%A2%E1%86%A8%20%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%89%E1%85%A6%E1%86%BA%20MIRACL.png?raw=true)
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/%E1%84%92%E1%85%A1%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8%E1%84%8B%E1%85%A5%20%E1%84%8B%E1%85%A5%E1%86%AB%E1%84%8B%E1%85%A5%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%87%E1%85%A6%E1%86%AB%E1%84%8E%E1%85%B5%E1%84%86%E1%85%A1%E1%84%8F%E1%85%B3%20LogicKor.png?raw=true)
    - 첫번째 이미지
        - 검색 모델: bge-m3
        - 다국어 문서 검색 데이터셋 MIRACL에서 가장 뛰어난 한국어 성능
    - 두번째 이미지
        - LLM: rtzr/ko-gemma-2-9b-it
        - 한국어 언어모델 다분야 사고력 벤치마크 LogicKor에서 오프소스 모델 중 10B이하 가장 높은 순위

*비교 실험*
- zero shot으로 요약한 경우:  23.932점
- 랜덤으로 train 데이터 예시를 넣어, few shot learning으로 요약한 경우: 29.312점
- test 데이터와 유사한 train 데이터를 예시로 넣어, few shot learning을 한 경우 31.305점

*문서 검색은 요약해야할 test 데이터의 dialogue와 train 데이터의 dialogue를 bge-m3 모델로 벡터화한 후 코사인 유사도로 가장 유사한 데이터를 검색하였습니다.*

*결론*
- test 데이터와 유사한 train 데이터를 검색 후 예시로 넣어 few shot learning을 하였을 때 가장 좋은 결과 나타남
- 전체적으로 보았을 때 좋은 점수는 아니지만, llm에서 의도하는 데이터를 통해 few shot learning을 적용했을 때 효과 확인

#### 장재성
*아이디어*
- Train Dataset과 Test Dataset이 영어를 번역한 번역체라는 점으로 미루어 보았을 때 원문이 영어일 가능성
- 원문이 영어였다면 영어로 학습시키면 좀 더 좋은 성능을 가져올 수 있지 않을까?
- 따라서 모든 Train 및 Test 데이터를 영어로 번역, 영어 모델을 사용하여 Output 생성 후 다시 한글로 번역?

*실험순서*
- Train 및 Tset 데이터를 한국어에서 영어로 번역
- 영어 모델 선정 및 코드 수정
- 해당 모델로 생성된 Output을 다시 한국어로 번역하여 제출

*Translator 선정*
- Google Translator, DeepL, Microsoft Translator
- 이 중 DeepL은 무료 버전이 있지만 Train Data가 12000여개에 달하는 양으로 제한에 걸릴 가능성이 높아 제외
- Google과 Microsoft 중 Google 번역기를 자주 사용했고 성능을 알기에 Google로 먼저 진행

*진행 사항 및 문제점*
- 12000여개에 달하는 Train Data는 Dialogue, Summary, Topic 총 3가지로 이루어져 있어 36000여개의 데이터가 있고 1개의 Data를 해석하는데 약 10시간이 걸림.
-  실제 Dialogue, Summary는 최소 3초 이상 소요되어 전체 번역하는데 하루 소모
- 처음에 번역이 잘 되는 것을 확인 후 전체 코드를 작성하였고 여기서 실수를 해 정확한 번역 데이터가 나오는 데는 반나절이 더 소모

*LLM 영어 모델 선정*
- Bart, T5, Pegasus
    - Bart : baseline에서 주어진 모델 또한 Bart에서 파생되어 1순위로 선정
    - T5 : 성명기 팀원이 좋은 성능을 이끌어낸 모델로 2순위 선정
    - Pegasus : Translator가 Google인 점, 다만 논문, 뉴스 기사 등 긴 글에 특화된 점으로 3순위 선정

*결과*

![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/(%E1%84%8C%E1%85%A2%E1%84%89%E1%85%A5%E1%86%BC)%E1%84%87%E1%85%A5%E1%86%AB%E1%84%8B%E1%85%A7%E1%86%A8%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%E1%84%8C%E1%85%A6%E1%84%8E%E1%85%AE%E1%86%AF.png?raw=true)


## 5. Result

### Leader Board
![](https://github.com/SUNGMYEONGGI/image/blob/main/Upstage-NLP-Project_Image/final%20score.png?raw=true)

### Presentation

- [Presentation](https://drive.google.com/file/d/1XzhaUBO_9jicPUHMY99QXxmSpJdDvSWr/view)
