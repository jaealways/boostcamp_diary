# 1 NLP Overview

## 대회 및 일정 소개

- 신경망 기반 NLP 개요
- NLP 도구 소개
- NLP 구현

## NLP Tasks

### NLP in Real Life [05:50]

- NLP가 가능한 일상 예시들 설명
- 트위터를 통해 재난상황인지 분류하는 task
- 텍스트 입력시 생물의학 개체 타입 추출
- AI 경진대회에서 코드 유사성 판단
- 이미지 입력했을 때, 이미지 설명하는 자연어로 아웃풋 낼 수도 있음

### NLP Tasks [10:35]

- 아래와 같이 다양한 NLP 태스크가 존재함
- 주제분류: 문장 들어왔을 때 주제 라벨
- 감성분석: 문장 들어왔을 때 긍부정 분류
- 개체명분석
- 형태소분석
- 대화
- 번역: N to N으로 번역

### Sequence to Sequence Learning [13:10]

- 복잡한 자연어 문제를 어떻게 Neural Net 기반으로 쉽게 풀까?
- Seq2Seq Learning 활용 → N개의 입력과 M개의 아웃풋 사이의 연관성을 파악하는 방법론
- N to N, N to M, N to 1 등 여러 형태로 나눌 수 있음

### Sequence to Sequence Learning for NLP Problems [17:40]

- N21: Classification, 감성분석
- N2N: 형태소분석 등 input과 정확히 매칭되는 무엇인가 내놓을 때
- N2M: 기계번역 등, Encoder와 Decoder로 역할 분류
- 본 강의에선 위의 예시들을 트랜스포머 기반의 모델로 설명하고자 함

## N21 Problem [20:50]

- 여러 개의 토큰 인풋에서 단 하나의 클래스 아웃풋

### Topic Classification [21:15]

- <CLS> 토큰  앞에 놓고 인코더에 넣음

### Semantic Textual Similarity [22:50]

- 문장 여러 개가 얼마나 유사한지 판별
- <SEP> 토큰을 통해 문장 사이 구별
- 숫자 아웃풋을 받을 수 있게 <CLS> 토큰 넣어줌

### Natural Language Inference [25:15]

- 가설 문장과 전제 문장의 관게를 추론
- 진실(entailment), 거짓(contradiction), 미결정(neutral) 등 multi label classification 문제로 바꿔서 해결

## N2N Problem [26:35]

- 보통 2~3개의 문장 인풋 → 이에 대응하는 classification을 아웃풋으로

### Named Entity Recognition [27:55]

- 많이 사용되는 개체명(사람, 위치, 조직 등) 미리 정의

### Morphology Analysis [30:25]

- 형태소 분석기, 각 입력 토큰에 해당하는 형태소 아웃풋으로 출력

## N2M Problem [31:40]

- N2M 기반으로 N21, N2N 등으로 확장 가능
- 자연어 생성이 중요할 경우, N2M 사용

### Machine Translation [34:50]

- Encoder 학습 후 <Start> 토큰과 함께 Decoder를 통해 단어 생성
- Output을 디코더의 인풋으로 넣고 Encoder에 실행을 반복 → <End> 토큰 나올 때까지

### Dialogue Model [36:50]

- 인코딩과 디코딩을 여러 단계에 걸쳐서 하면서, 다음 텍스트를 예측

### Summarization [37:40]

- 추상화기법(인코더 디코더 사용)을 사용해서 요약하면, 본문 외의 내용도 등장 가능

### Image Captioning [40:45]

- 이미지를 인풋으로 하고, 텍스트를 아웃풋으로 함
- Seq2seq2 learning의 인코더와 디코더가 조금만 바뀌면 가능해짐
- 이미지를 vector space에 위치시킴, 해당 vector space 정보를 바탕으로 Decoder 시작

### NLP Benchmarks and Problem Types [42:45]

- GLUE, KLUE 등의 데이터셋은 각각 N21에서 N2M까지 다양한 태스크를 처리할 수 있음

# 2 PyTorch Lightning 이론

### Deep Learning Blocks [01:15~]

딥러닝의 진행단계

Deep Learning Process 

데이터 전처리 → 신경망 디자인 → output 활용, 후처리

데이터 전처리/준비단계 [03:08~]

- data Preparation -
    - 데이터 불러오기!
    - 데이터 분절화 - 8:2 로 train,Vaild,Test 데이터로 자르기
    - Prepare Data Feeding function → 데이터들을 텐서화 시키기
    - Make Batch data → 배치로 데이터 묶어주기
- model Implementation - 모델 구현
- Loss Implementation - 모델이 얼마나 틀렸는지를 측정하기 위한 함수설정
- Updater Implementation - 옵티마이저 설정
- Iterative Learning - 데이터 Feeding 을 통한 모델 반복 학습 및 검증

### PyTorch Lightning [22:20 ~]

운전할때 엔진운동방식 알 필요없듯이 우리도 개어려운 딥러닝 다 알필요없다! 인터페이스만 다룰 줄 알면된다!

- Keras 위와 같 은 의미에서 텐서플로를 사용하기 편한 인터페이스를 만드려고 한 시도다!
- PyTorch Lightning 은 케라스의 파이토치 버전. High-level 인터페이스 제공, 높은 수준의 자동화!
    - Data Preparation: PyTorchLightning.LightningDataModule
    - Model Implementation, Loss Implementation, Updater Implementation: PyTorchLightning.LightningModule
    - Iterative Learning: PyTorchLightning.Trainer

### Data Preparation [34:01~]
- PyTorch Lightning이 조금 더 생각의 흐름을 잘 반영함 -> 아니면 if then 자주 써서 코드 지저분해짐
- prepare_data(): 데이터 다운로드, process에서 한 번 호출됨
- setup(): 데이터 분할, gpu마다 호출하여 처리
- batch로 인해 코드 복잡해지는데, DataLoader로 들고다닐 필요 없어짐 -> 프로그래밍 낭비 줄일 수 있음

### Model Implementation [38:20]
- nn.Module 대신 pl.LightningModule로 바꿔주면 됨

### Optimizer [39:25]
- PL: configure_optimizers() 안에 생성하고 Train step이 끝날 때마다 자동호출

### Train & Loss [40:10]
- PL: Train & Loss를 step에서 진행, 진행 사항을 자동으로 출력


### Validation & Loss [41:15]
- PL: Validation & Loss를 step에서 진행, 진행 사항을 자동으로 출력

### Test & Loss [41:20]
- Test & Accuracy at step 결과를 자동으로 출력, 개발속도가 빨라짐
- 매번 반복되는 코드를 구조화시켜서 구현 가능


Plus! 다른 API 들 소개 [44:55~]
- TorchMetrics
    - GPU 기반의 metrics를 사용할 수 있음


# 2-실습 PyTorch Lightning 실습
- DataLoader에서 shuffle True 중요
- net.eval을 명시적으로 언급해주기(train 단계냐 eval 단계냐 dropout 등에서 중요)
- with torch.no_grad() -> 내 코드가 파라미터 업데이트 대상이 아님을 알림


# 3 NL Data 관리 및 처리 툴 소개

## 1. Pandas [01:40]

- GB단위 이상의 대용량 데이터 처리 용이
- 테이블과 시계열 조작하기 위한 데이터 구조 및 연산 제공
- “액셀”과 비슷한 형태의 자료구조들을 지원 [02:30]
- Series와 DataFrame [03:15]
- 데이터 선택 [06:15]
- 데이터 분석 [09:05]
- 그 외 함수들 [09:50]

## 2. Pandas with NLP [10:40]

- 데이터 읽고, 쓰기 - 매우 많은 형태의 데이터를 지원(Excel, CSV, TSV, JSON, XML 등)
- 데이터 탐색 [13:30]
- 데이터 분석 [16:35]

## 3. 실습 코드 [18:10]

- 필수 라이브러리 설치
- 데이터 탐색
- 데이터 분석PyTorch Dataset

# 4 Tokenization

### Tokenization이란? [2:45 ~]

- BoW (Bag of Words) : 오랜 전통 방식
→ 단어 가방에 담긴 단어 중 나타난 단어의 빈도를 벡터로 전달.
- TF-IDF [5:10 ~]
→ 단어 빈도 + 역문서 빈도를 사용하여 가중치 변환 ( 내 문서에 많이 나타난 단어 중요 )
- Word2Vec [6:30 ~]
    - 신경망 기반 방법론이 발달한 계기는, 신경망에 어울리는 임베딩이 가능해졌기 때문
→ A B C 단어가 나타날 때 B 단어는 A, C라는 단어와 연관있는 단어.

- Text를 숫자로 바꾸기 위해선 Tokenization + Embedding이 되어야 함
    - Tokenization: Text를 어떤 token 단위로 나눌 것인가?
    - Embedding: token을 어떤 숫자로 바꿀 것인가?


### Tokenization Methods [9:45 ~]

- Tokenization Methods | Examples [10:15 ~]
- Character-based Tokenization [13:35 ~]
’space’도 하나의 토큰으로 고려
→ 긴 length가 많들어지므로 많은 계산량과 메모리 필요 (컴퓨터에 부담, 사람에게 편리)
- Word-based Tokenization [14:45 ~]
’space’에 따라 단어 토큰화
→ 언어에 따라 띄어쓰기가 효율적이지 않은 경우도 많음 (특히 한국어)
- Subword-based Tokenization [15:40 ~]
문장 혹은 단어를 의미있는 단위로 묶거나 분할해서 처리
→ ‘tokenization’ = ‘token’ + ‘-ization’
- Byte Pair Encoding (BPE) [17:45 ~]
    어휘를 묶어나가는 알고리즘
        
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/d971c4fd-ba0f-4f04-af82-61c01335868c/02a3428e-5996-49ba-8502-899cd685fea6/Untitled.png)
        
10회 반복하며 es, est, lo 등 긴 단어의 집합이 추가됨.
        

### 한국어 토큰화 툴 [21:15 ~]

- NLPy - 한국어 토큰화 [21:20 ~]
- SentencePiece - English and other [22:05 ~]
- Haggingface-Tokenizer - 이미 학습된 tokenizer를 api로 공유 [22:50 ~]


# 5 Transformer

## 1.1 기존 NN의 문제점 [01:49]

1. FCN : 1개의 input → 1개의 output 의 문제점
2. RNN : 이전 정보를 기반으로 다음 정보 파악
3. RNN의 문제점
    - 순차적 모델링 방식
    - 지역 정보만 활용
4. Transformer
    - multiple items: 여러 개 토큰을 어떻게 잘 인코딩 할 수 있을까
    - long-term dependency
    - sequential information: 순서 정보를 어떻게 잘 처리(보존)할 수 있을까
    - fast: 빠르게 할 수 없나?
    - simple architecture

## 1.2 Attention Machanism [08:55]

1. Sequence를 Blending
    - 직관적 의미
        - 반응도가 물질 위주로 합성
        - 자연어, 이미지 등 가능
2. Machanism
    - scaled dot
    - query - key - value
    - blending score
    - multi-head attention : 다각도 해석

## 1.3 Transformer [20:50]

1. Transformer Architecture
    - encoder
        - multi-head (self) attention
        - add & norm
        - feed forward
    - decoder
        - masked multi-head (self) attention
        - multi-head attention : decoder의 q & encoder의 v, k
        - (decoder only block)
        - add & norm
        - feed forward
    - N21, N2N, N2M(encoder-decoder)


# 7-1 N21(1)

## Neural Network based Classification

### 접근방법 [01:40]
- 대부분 아래 두 가지 방식으로 접근
- Supervised Learning: 문제집 풀어보면서 답 잘 맞추게 하는 방식
- Classification: 사전에 클래스 갯수가 정해져야 함, 객관식 문제처럼 가장 높은 확률 부여된 정답 고르기

### 감성 분류기 구현 예 [06:40]
- N21 문제, 감성분석기를 구체적으로 구현해보고자 함
- 긍정, 부정, 중립, 객관의 4개 클래스를 가정
- 각 문장마다 클래스를 사람이 할당해 줌

### 신경망 디자인 [09:30]
- 4개 클래스가 정의된 Network 정의하기
- 인풋 -> 네트워크를 통해 계산 -> 클래스별 scoring
- 5개 입력 토큰 -> 4개 class 아웃풋, 총 24(bias 포함)개의 파라미터 필요

### 신경망 분류 기법 [13:50]
- 신경망을 통해 분류 진행할 때 네 가지 하위 문제
    - Reference Representation
    - Scoring Normalization
    - Cost Function Design
    - Parameter Update

# 7-2 N21(2)

## 1.4 신경망 분류 기법 [00:00]

- 정답 표현
    - Tokenization과 Embedding을 기존에 썼음
    - 정답을 어떻게 표현할 수 있을까?
        - 정답 데이터의 일관성이 중요함 [03:30]
        - classification에서 여러 감정 불가, 감정의 정도를 따지지 않음
        - 일관성을 잘 표현한 One-Hot Representation의 장단점
- Score Normalization [07:00]
    - 예측값과 정답을 서로 비교하려면 같은 Scale 값이어야 함[08:20]
- Softmax
    - exponential 함수의 특징 - 무조건 0보다 크고, 기하급수적으로 올라감 [09:10]
    - 전체 점수에서의 일정 비율이 나오게 됨 - 전체 합 1.0 [11:00]
    - 정리 [11:55]
- Cost Function [12:25]
    - Loss Function, Cost Function, Objective Function 등은 비슷한 표현
    - 단 cost function은 미분 가능해야
    - 두 점수들 간의 차이를 수치화
    - Cross Entropy에 대해 [16:00]
    - 방향성이 있는 확률분포 간의 비교 (KL divergence)
    - 방향성을 따지지 않을 때 (Cross Entropy)
- Parameter Update [19:00]
    - 오류가 작아지는 방향으로! [19:35]
    - 오류가 작아지는 방향을 결정하는 방법(Gradient Descent) [20:00]
- 총 정리 [21:20]


# 마스터클래스
- 정보에 접근, 생산하는 방식이 기하급수적으로 증가
- 100년 전만 해도, 고령자의 넓은 인지범위로 좋은 의사결정함
- representation paradigm이 변하고 있음
- symbol이 갖는게 인간의 능력(지능) 그 자체라고 할 수 있음
    - 그림 -> 언어 -> 음표, 화학식 등
- 뉴럴넷은 이 모든 걸 다른 방식으로 표현
- 인공지능 내에서의 onthology, logic을 규현해왔음 (심볼기반) -> 손바닥, 주먹 등이 아니라 다른 손 모양은 구체적으로 기호화할 수 없음
- 현실의 심볼에서 구멍을 내서 학습 시킴 -> 이제 다양한 문제를 풀 수 있게 됨
- 최근의 트렌드 relational database -> vector database (숫자로)

