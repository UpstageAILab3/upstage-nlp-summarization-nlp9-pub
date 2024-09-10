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

### Overview
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
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
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

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
