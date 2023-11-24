체크리스트 Week3

- [ ]  cnn layer별로 필터 거치면서 차원 어떻게 변하나? 숙지하기
- [ ]  bottleneck 아키텍처의 필요성?
- [ ]  5강 resnet [19:40] 부근에 차이만 컨볼루션 레이어가 학습한다는 의미?
- [ ]  Resnet skip-connection 의미?: 그래디언트가 뒤로 직접 전달되게 해서 소실문제를 해결?
- [ ]  퀴즈7 2번. RNN은 모든 시간 단계에서 동일한 가중치를 사용?
- [ ]  Multi-Head Attention에 대한 추가학습(어떤 프로세스 거치는지, 차원 어떻게 변하는지)
- [ ]  트랜스포머 masking: 이전 단어들만 dependent하게 하기 위해서?
- [ ]  self attention은 순서에 independent하게 각 단어에 임베딩함!
- [ ]  9-1강 17분 50초 근처에 마르코프 성질 적용하면 왜 2n-1?

# 3.1 Historical Review

## Introduction

### 딥러닝의 개념 [00:35~]

- 딥러닝은 매우 광범위한 개념.
- 좋은 딥러너가 되기 위해선?
    - 구현 스킬, 수학 스킬(선형대수, 확률론), 최신 논문 많이 알기
- 딥러닝, 머신러닝 범주화해서 비교

## 딥러닝의 주요 요소

### 딥러닝 주요 요소 소개 [05:50~]

- Data, Model, Loss, Algorithm
- 논문 볼 때, 위의 네 가지 잘 살펴보면 구체적인 contribution 살펴볼 수 있음

### Data [07:50~]

- 어떤 문제를 푸느냐에 따라 필요한 데이터 다름

### Model [09:05~]

- 데이터를 풀고 싶은 문제의 형태로 바꿔주는 역할

### Loss Function [09:50~]

- 모델과 데이터가 정해져있을 때, 어떻게 학습할 것인가와 관련
- 어떤 문제 (Regression, 분류 등)를 푸느냐에 따라 다른 식 세워짐

### Optimization Algorithm [12:20~]

- 모델, 데이터, Loss Function이 정해져있을 때 Network를 어떻게 줄일까?
- Dropout, Early stopping 등 다양한 테크닉이 있다
- Loss function을 단순히 줄이는게 중요한게 아니라, 학습되지 않은 데이터에서 잘 하는게 중요

## 딥러닝 Historical Review

### 딥러닝의 역사 [14:00~]

### AlexNet 2012 [15:20~]

- ImageNet 대회에서 딥러닝 기반으로 압도적 성능 보임

### DQN 2013 [16:55~]

- 게임에 강화학습 사용

### Encoder/Decoder 2014 [18:00~]

- Neural Machine Translation 문제 푸는데 사용

### Adam 2014 [19:05~]

- Adam이 결과 잘 나옴, 그냥 사용하는 경우 많음

### GAN 2015 [22:00~]

- Generative Adversarial Network
- Generator, Discriminator로 학습

### ResNet 2015 [23:35~]

- Layer를 매우 깊게 쌓아도, 성능(테스트데이터셋)을 보존해줌

### Transformer 2017 [25:20~]

- Attention 구조, 매우 중요

### BERT 2018 [27:10~]

- Bidirectional Encoder Representations from Transformers
- Fine-Tuning NLP가 발전하는 계기

### Big Language Models 2019 [28:35~]

- GPT-3, 파인튜닝 모델의 끝판왕 등장

### Self-Supervised Learning 2020 [29:25~]

- SimCLR: 라벨을 모르는 데이터를 학습에 같이 활용
    - 이미지에 대한 좋은 representation을 학습 데이터 외의 데이터로 추가 학습하고자 함
- 도메인 지식 많이 있으면, 학습 데이터를 추가로 만드는 것도 하나의 트렌드

# 3.2 Neural Networks & Multi-Layer Perceptron

## Neural Networks

### Neural Network의 정의 [00:45~]

- 뇌의 뉴런을 모방한 네트워크
- 날고 싶다고 새와 꼭 닮을 필요가 없듯이, 뉴럴넷도 인간의 뇌 모양에만 국한될 필요 없음
- 목표함수로 근사하는 function approximator라고 새롭게 정의할 수 있음
    - 행렬 곱 + 비선형변환을 반복적으로 해서
- GoogLeNet, ResNet 등으로 설명

## Linear Neural Networks

### Linear Neural Networks [06:35~]

- 가장 간단한 뉴럴넷은 사실 선형함수임
- 선형모형에서 Data, Model, Loss의 정의

### 변수 최적화 [08:50~]

- 편미분 사용해서 변수 최적화 (수식설명)
- u, b 등의 변수 업데이트
- stepzise 에타 설명

### Multi Dimensional [13:05~]

- 다차원에서 행렬 사용해서 아웃풋 구할 수 있음
- 행렬 연산이 어떻게 일어나는지에 대한 시각적 설명

### Beyond Linear Neural Networks [14:45~]

- 여러 개의 뉴럴넷을 층층히 쌓을 때, 비선형변환 필요
- 여러 Activation Function 소개

### Universial Approximation Theorem [16:55~]

- 뉴럴넷이 잘 되는 또 다른 이유
- Hidden Layer가 하나만 있어도, 대부분의 연속 함수 근사시킬 수 있음
- 뉴럴넷의 표현력이 좋다는 의미이지, 내 함수의 성능이 반드시 좋다는 보장은 아님!

## Multi-Layer Perceptron

### Multi-Layer Perceptron [18:35~]

- 다층 구조의 퍼셉트론 개념(이미지) 소개

### Loss Function [19:15~]

- Regression Task일 때
- Classification Task일 때
- Probabilistic Task일 때

# 3.3 Optimization

- 최적화의 여러 용어들의 개념을 확실히 집고 넘어가는게 좋음

## Important Concept

- Generalization [03:41]
    - 많은 경우 일반화 성능 높이는게 좋음
    - 학습데이터의 성능이 매우 안좋으면, generalization(test와 train의 차이) 퍼포먼스가 좋다고 test 성능이 좋지는 못함
- Under-fitting vs Over-fitting [05:16]
- Cross Validation [06:20]
    - 하이퍼파라미터와 파라미터 간에 어떤게 좋고 안좋은지 모르니까, cross-validation을 통해 최적의 하이퍼파라미터 셋을 찾고 이를 바탕으로 학습 시킴
- Bias vs Variance [09:08]
    - Variance: 출력의 정도가 얼마나 퍼지는지
    - Bias: 평균적으로 (군집이) 정답에서 얼마나 벗어나는지
    - Bias & Variance & Noise Tradeoff [10:26]
        - Bias와 Bariance는 Tradeoff 관계
        - cost=bias^2+variance+noise
- Bootstrapping [11:49]
    - Booststraping: 학습데이터가 고정일 때, 여러 sub sampling 방법을 통해 학습하겠다
    - Bagging [13:17]
    - Boosting [14:34]
        - 여러 Week Learner들의 결과를 합침

## Gradient Descent Method

### Gradient Descent [16:11]

- Stochastic gradient descent
    - 한 번에 한 개만 구한다?
- Mini-batch gradient descent
    - 여러 번 batch로 나눠서 학습
- Batch gradient descent
    - 한 번에 10만개 다 사용

### Batch-size Matters [17:20]

- Flat Minimum
    - 일반화 성능이 높음
- Sharp Minimum
    - 일반화 성능 안좋을 수 있음
- Batch Size 줄이면 일반적으로 일반화 성능 좋음

### Gradient Descent Methods

- 각각의 Optimizer가 어떤 특징 있는지 알아보면 좋음
- (Stochastic) gradient descent [21:15]
    - Learning Rate, Stepsize 찾는게 너무 어려움
- Momentum [21:56]
    - beta가 momentum 잡음
    - 모멘텀과 현재 gradient가 포함된 accumulation으로 업데이트
- Nesterov Accerlerate [23:30]
    - Lookahead (해당 방향으로 한 번 가보고 Accumulate)
    - 봉우리에 더 빠르게 converge?
- Adagrad [25:20]
    - Ada: Adaptive
- Adadelta [27:20]
    - Adagrad가 갖는 $G_t$가 계속 커지는 현상을 막겠다
    - exponential moving average를 사용해서 막겠다
- RMSpop [29:29]
    - 많이 사용되었음. Adadelta에 에타라는 스텝사이즈 추가함
- Adam [30:30]
    - 가장 무난하게 사용
    - Adaptive Moment Estimation:

## Regularization

- 학습을 방해하도록 규제를 거는 것
- 학습 데이터 뿐만 아니라 테스트 데이터에서도 잘 동작하게 하는 것
- Early Stopping [32:30]
    - 학습을 일찍 멈춤, validation (학습데이터 제외) 사용
- Parameter norm penalty [34:18]
    - 뉴럴넷 파라미터가 너무 커지지 않게 하는 것
- Data augmentation [35:11]
    - 데이터가 어느정도 커지면 뉴럴넷 성능이 가장 압도적
    - 주어진 데이터를 지지고 볶아서 늘리고자 함
    - 숫자 데이터는 뒤집으면 안되지만, 비행기 자전거 등 분류는 가능
- Noise Robustness [37:27]
    - 입력 뿐만 아니라 weight에도 노이즈 집어넣음
- Label Smoothing [38:09]
    - Decision boundary를 부드럽게 만들어주는 효과
    - 들인 노력대비 성능 많이 높일 수 있음
- Dropout [40:24]
    - 각각의 뉴런들이 조금 더 robust한 feature 잡을 수 있음
- Batch Normalization [40:56]
    - 내가 적용하려는 layer의 statistics를 정규화시키는 것
    - Batch Norm, Layer Norm, Instance Norm Group Norm 각각의 장단점?
    

# DAY 11 회고

팀별 요약을 같이 한 것이 학습에 정말 많은 도움이 되었다.

네부캠의 커리큘럼이 굉장히 좋지만 개인적인 어려움이 있었다.

- 광범위한 진도
- 빠른 흐름

2주차 파이토치에서 큰 흐름을 많이 놓쳤던 것 같다.

한 눈에 흐름이 보이니 취사선택이 가능해졌다.

또한 필기에도 병목이 많이 줄었다.

사실 공수가 적게 드는 것도 아닌데, 흔쾌히 도전에 응해준 팀원들이 고마울 뿐이다.

# 3.4 Convolutional Neural Networks

## Convolution [00:00]

### Convolution 수식 [00:16]

- 두 개의 함수를 잘 섞어주는 operator로의 정의
- Continuous convolution
- Discrete convolution
- 2D image convolution
    - I = Image
    - K = Kernel = Filter

### Convolution 직관적 설명 [01:12]

- Convolution 계산 과정 [01:12]
- Convolution 의미 : 특징 추출 결과 [02:05]
    - Filter의 특성에 따라 (가령 3 by 3의 평균을 내면 blur되는 효과) 아웃풋이 달라짐

### RGB Image Convolution [03:06]

- #filter = 1 [03:06]
- #filter = n [03:44]
    - 필터의 갯수에 따라 아웃풋의 숫자가 달라짐

### Stack of Convolution [04:30]

- conv → activation(relu) → conv → activation(relu) → …
    - 각 필터의 차원이 어떻게 계산되는지 다시 한 번 복습하기

## Convolutional Neural Network [06:22]

- **convolution layer** : feature extraction (→ 여러 특징 추출을 위해 layer 증가)
- **pooling layer** : feature extraction
- **fully connected layer** : decision making (→ parameter 개수를 줄이기 위해 layer 감소)
- **fully connected layer를 줄이는 이유** [07:25]
    - #parameter(=parameter 개수) 줄이기 위함
    - #parameter 많다면?
        - 학습 어려움
        - generalization 잘되지 않음
    - GooLeNet의 #parameter [08:54]

## Stride, Padding [09:34]

- **Stride** [09:34]
    - 얼마나 dense하게 cn을 옮기나?
- **Padding** [10:55]
    - 가장자리 덧대주는 역할, dummy variable처럼?
- Stride, Padding 직관적 영상 [11:53]
    - 입력과 출력의 차원을 같게 하기 위한 padding 개수 [12:27]

## parameter 구하기 [13:50]

- AlexNet #Parameter 구하기 [16:15]
- Conv vs Dense layer의 #parameter가 차이나는 이유 [20:07]
    - CNN 일종의 shared parameter? 잘 이해 못함

## 1*1 convolution [21:39]

```
- 채널 차원 감소를 위함 → #parameter 감소 & #layer 증가
- ex. Resnet, Densenet
- bottleneck 아키텍처
    - 앞뒤로 1by1 넣으면 파라미터 줄일 수 있다...
- cnn에서 bias 고려하면 어디항에 +1?
    - 각 필터에 대한 파라미터 수는 (3*3*64=576), (576+1)*32

```

# 3.4-실습 Convolutional Neural Networks

## **Convolutional Neural Network (CNN) - MNIST dataset**

### 1. import, dataset, data loader (이전 실습과 동일)

### 2. model class 생성 [01:10]

- init
    - convolution layer [02:05]
        - convolution
        - batch-norm
        - max-pooling
        - dropout
    - dense layer [04:14]
        - flatten
        - linear
        - relu
    - network [05:19]
        - sequential
        - add_module (layer 이름 할당 가능)
- init param [05:45]
- forward [06:16]
- layer 별로 이름 정하면, 디버깅할 때도 훨씬 이해하기 쉽다

### 3. model, loss function, optimizer 생성 [07:23]

### 4. check parameters [08:25]

### 5. forward (랜덤 image 2장 모델에 돌려보기) [10:10]

### 6. evaluation (이전 실습과 동일) [11:03]

### 7. train (이전 실습과 동일) [11:23]

- model.train() vs model.eval() [11:54]
- 결과 분석 [12:40]

### 8. test (이전 실습과 동일) [13:20]

### 9. network tuning을 위한 팁 : layer 매개변수 [14:22]

# 3.5 Modern Convolutional Neural Networks

- 네트워크의 뎁스는 점점 깊어지고, 파라미터는 줄고, 성능은 향상됨

### AlexNEt [00:42, 03:57~]

- AlexNet은 네트워크가 두 개로 나눠져있음, GPU가 부족해서 GPU를 최대한 활용하면서 파라미터 많이 하기 위해서 사용한 전략?
- 11*11 filter를 사용하는 건 그닥 좋은 전략 아님, 채널이 1이면 121개 파라미터 필요...
- 총 8단 뉴럴넷
- AlexNet의 성공이유
    - ReLU 사용
    - 2개 GPU 사용
    - Local response normalization, overlapping pooling 사용
    - Data augmentaion 사용
    - Dropout 활용
- Activation function 사용하면 vanishing gradient 문제 생김 -> ReLU로 해결

### ILSVRC - Imagenet LArge-Scale Visual Recognition Challenge [01:31~]

- 사람이 error rate 5.1%인데, 15년에 이미 초월함

### VGGNet [08:47~]

- 3x3만을 활용함 왜??
    - 컨볼루션의 사이즈가 커지면 같는 장점: 한 번 찍었을 때 고려되는 인풋의 크기가 커짐 (Receptive Field)
- 3x3 두 층이 5x5 하나보다 파라미터 더 줄일 수 있음

### GoogLeNet [12:08~]

- 1 by 1 컨볼루션은 dimension reduction 효과가 있음 -> 어떻게 잘 활용할지가 GoogLeNet이 주는 레슨
- Inception block의 정확한 의미?: 1 by 1 컨볼루션 덕분에 파라미터 줄이는 효과

### Inception Block [14:20~]

### 갑분퀴즈 - 어떤 CNN 아키텍처가 가장 적은 수의 파라미터를 갖고 있나요? [18:38~]

답 - 구글넷

### ResNet [19:40~]

- 파라미터 숫자가 많으면 오버피팅 일어날 수 있음 (트레이닝 에러는 주는데 테스트 에러는 오히려 커짐)
- 컨볼루션 레이어가 학습하고자 하는 것은 차이만(?)
- ResNet 구조를 사용하면, 훨씬 깊게 쌓아도 학습을 시킬 수 있다!
- prokected shortcut: 1by1 conv 사용해서 차원 맞춘 후 덧셈 연산 가능하도록 해줌
- batch norm을 relu 뒤어 넣는 것이 더 잘됨? 오히려 안 넣는게 더 잘됨? 약간의 논란 있음
- bottleneck 아키텍처: 1by1 채널을 앞뒤로 넣어주어서 원하는 차원 맞출 수 있음 (조금 더 직관적으로 이해하고 싶음)

### DenseNet [25:15~]

- 컨볼루션을 통해 나온 값을 더하지말고, concat하자!
- 채널이 기하급수적으로 커짐! 파라미터도 기하급수적으로 커짐!
- 중간에 채널을 한 번씩 1 by 1 conv로 줄임
- DenseNet이 SOTA 차지하는 경우 많음

### Summary - 최종정리

VCG : 3x3 블록의 반복

GooLeNet : 1X1 convolution

ResNet : skip - connection

DenseNet : concatenation (연쇄)

# 3.6 Computer Vision Applications

## semantic Segmentation [00:50~]

- 이미지의 각 픽셀을 라벨마다 분류함
- 자율주행 등 개별 사물을 인식하는데 활용

### Fully Convolutional Network [03:05~]

- CNN은 conv, pooling, dense 등을 통과하다가 마지막에 fully-connect layer를 통과해서 결과 내는 방식이었음
- Fully Convolutional Network는 dense layer를 없애자는 것! (convolutionalization)
- 쭉 펴서 계산하자는 의미로 받아들일 수 있음
- 파라미터 숫자도 같은데 왜 semantic segmentation에서 convolutionalization를 할까?
- convolution이 갖는 shared parameter 성질 덕분에, 동일한 convolution 필터가 동일하게 찍기 때문에 resulting space 차원이 유동적으로 커졌다 줄었다 할 수 있음
- 즉 차원이 무수히 커져도 무사히 CNN을 동작시킬 수 있음 + 히트맵처럼 정도를 표기할 수 있음
- 아웃풋을 원래의 dense pixel로 늘리는 역할 필요. unpooling 등 다양한 기법 사용

### Deconvolution(conv transpose) [09:55~]

- convolution의 역연산(으로 생각하면 좋음)
- 엄밀하게 convolution을 복원하는건 불가능
    - 2+8, 3+7 모두 10, 10에서 역으로 가면 여러 경우의 수 나옴
- 그림을 통한 Deconvolution의 이해

### results [12:10~]

- 인풋이미지에서 fully convolution network에서 (??)

## Detection (탐지 - 분류) [12:50~]

- 물체를 분류하는데, bounding box로 하는 것
    - 이미지에서 2000개의 region을 뽑음
    - region의 크기를 똑같이 맞춘다음, feature를 CNN으로 추출
    - linear SVM으로 출력

### R-CNN [14:10~]

- 이미지 안에 어떤 물체가 있는지 구할 수 있음

### SPPNet [14:45~]

- R-CNN의 가장 큰 문제는 2000개의 region을 뽑으면, 2000개의 이미지를 모두 통과시켜야 함
- SPPNet은 이미지에서 바운딩 박스 뽑고, 이미지 전체에서 convolution feature map 만든 다음, 뽑힌 바운딩 박스 위치에 해당하는 텐서만 가져오자!
- R-CNN에 비해 속도가 빨라짐
- spatial pyramid pooling: 가져온 feature map을 어떻게든 하나의 벡터로 바꿔줌

### Fast R-CNN [16:20~]

- 기본 컨셉은 SPPNet과 동일. 뒷단에 뉴럴넷을 통해 bounding-box regression과 classification을 하는 점에서 차이

### Faster R-CNN [17:40~]

- 이미지에서 바운딩 박스를 뽑아내는 region proposal도 학습을 통해 하자!
- Fast R-CNN + Region Proposal Network

### Region Proposal Network [18:30~]

- 이미지에서 특정 영역에 물체가 있을지 없을지 찾아주는 역할 수행
- 물체가 무엇인지는 뒷단의 네트워크가 판단
- Anchor Box: 물체의 크기에 맞는 탬플릿을 미리 예측해 놓음!
- Fully Conv Net이 해당 영역에 물체 있을지 없을지 정보 갖게 됨
- 필요한 파라미터 갯수: (서로 다른 region size x 서로 다른 region 비율) x (바운딩 박스의 변형에 대한 파라미터 + 박스가 필요한지)

### YOLO [21:40~]

- 이미지 한 장에서 바로 아웃풋이 나올 수 있기 때문에, Faster R-CNN 보다도 훨씬 빠름
    - 이미지를 S by S 그리드로 나눔
    - 그리드 중간에 찾고자 하는 물체가 들어가면, 바운딩 박스와 해당 물체 동시에 예측해줌
- 최근의 Detection 모델들은 바운딩 박스 사이즈를 미리 정해놓고, 얼마나 변형시켜야 할지의 문제로 바꿈

# DAY 12 회고

체력 관리가 중요하다. 아침에 에너지를 너무 쏟다가, 피어세션이 끝날 때 쯤이면 기진맥진하는데 에너지를 잘 나누어 쏟을 방법을 고민해야겠다.

# 3.7 Recurrent Neural Networks

## Sequential Model [01:10]

- 시퀀셜 데이터를 처리할 때 가장 어려운 것은 길이가 언제 끝날지 모름. 차원을 예측할 수 없기 때문에 fully connected layer 사용 못함
- 다음번 입력에 대한 예측을 하는 것을 예로 들 수 있음
- 고려해야 할 contditioning vector 숫자가 점점 늘어남
- Fix the past timespan(과거의 몇개만 보기)
- Markov model - 나의 현재는 바로 전 과거에만 dependent함
    - joint distribution 표현하는건 쉬워지지만, 많은 정보를 버리게 됨
- Latent autoregressive model
    - hidden state가 과거의 정보를 요약해서 가짐
    - 현재 입력이 이 hidden state에만 dependent

## Recurrent Neural network [07:00]

- RNN - 시간 순으로 푼다고 이야기를 많이 함
- 멀리 있는 정보가 살아남기 힘듦
- Short-term dependencies
- Long-term dependencies
    - RNN이 포착하기 힘듦
- Vanishing / exploding gradient 문제
    - sigmoid 사용하면 vanishing, ReLU 사용하면 exploding 됨
- Long Short Term Memory(LSTM)의 등장

## Long Short Term Memory(LSTM) [13:20]

- LSTM 구조
    - Input
    - Output(hidden state)
    - Previous cell state
    - Previous hidden state
    - Next cell state
    - Next hidden state
- gate 위주로 이해하기 (컨베이어 밸트에서 어떤거 버릴지 정하기)
    - Forget gate - 어떤 정보를 버릴지 결정
    - Input gate - 어떤 정보를 cell state에 저장할 지 결정
    - Update cell - 위 정보를 통해 cell state 업데이트
    - Output gate - Update cell state를 통해 output 생성

## Gated Recurrent Unit(GRU) [22:40]

- reset gate(forget gate랑 비슷), update gate 두 개로만 심플하게 구성
- No cell state, just hidden state
    - hidden state가 곧 output이고 바로 다음번 gate로 들어감
- LSTM과 비슷한 역할이지만 보통 성능 더 좋음
- 요즘에는 LSTM, GRU 다 잘 안씀(Transformer가 나오면서 거기로 넘어감)

# 3.7-실습 Recurrent Neural Networks 실습

## Classification With LSTM [00:00]

- sequential 데이터는 전처리에 공수가 많이 필요해서 MNIST를 사용!
- Dataset and Loader - sequential data가 필요 [02:25]
    - MLIST dataset 이용
    - 최종적으로, batch 마다 one-hot vector가 튀어나와서 argmax를 통해 확률로 얻어짐
- Define Model [02:30]
- Check How LSTM Works [05:30]
- Check parameters [08:10]
    - 파라미터가 생각보다 많음(여기 82만 개)
    - 각각의 gate function이 모두 dense layer임
    - 파라미터를 줄이기 위해 hidden dimension을 줄여야 함
- simple Forward Path [09:15]
- Evaluation Function [09:25]
- Initial Evaluation [10:00]
- Train [10:10]
- Test [12:30]

# 3.8 Transformer

## Sequential Model이 왜 어려운지 [00:10]

- Trimmed sequence
- Omitted sequence
- Permuted sequence
- 중간에 단어가 빠지기도 하고, 뒤바뀌기도 하는 등 언어 모델의 sequence가 규칙적이지 않음

## Transformer [01:45]

- 재귀적인 구조 없이, attention이라는 구조를 활용 [01:55]
- Sequential한 데이터를 처리하고 인코딩 → NMP뿐 아니라 분류, 인식도 가능 [03:00]
- 하려는 것 : 변환 (예 : 프랑스어 → 영어) [04:55]
- 입력과 출력 시퀀스 개수가 다를 수 있음 [05:25]
    - n개의 단어가 어떻게 인코더에 한 번에 처리 될 수 있는가?
    - 인코더와 디코더에서 어떤 정보를 주고 받는가?
    - 디코더가 어떻게 generation을 할 수 있는가?
- Encoder = Self-Attention + Feed Forward Neural network [07:25]
    - Self-Attention이 중요
- 처리 방식 [08:25]
    - 단어를 벡터로
    - Self-Attention / 하나의 단어 씩 수행 [08:50]
        - 진행 시, 모든 n개의 단어와의 dependency를 고려
    - Feed Forward neural Network
- Self-Attention at a high level [10:30]
    - 문장에서 it의 의미 파악하기 (하나의 단어가 아니라 문장에서 어떤 관계를 갖는지 잘 파악해야)
    - it이 animal과 높은 관계를 가짐을 알아서 학습함
    - 주어진 단어 하나 당 3가지 벡터 Query(q), Key(j), Value(v) 생성 [11:15]
    - 단어 하나 Encoding 예시 [11:55]
        1. Score(유사도 계산) = q와 k 내적(스칼라) [13:30]
            1. 자기 자신의 q와, 모든 단어들과의 k를 내적(자기 자신 포함)
            2. 이 결과를 통해, 각 time step마다 어떤 단어와 interaction을 많이 해야 할 지가 정해짐
        2. Score들을 쭉 나열해서, 각각을 Nomalize하고 softmax 취해주기 → 각 Attention Weight 탄생(스칼라) [14:10]
        3. Attention Weight를 나열한 Attention(벡터)과 Value(벡터) weighted sum 진행
        4. Value와 Attention 가중합 → sum(최종 인코딩 벡터) 탄생
    - q, k 벡터는 사이즈 같아야 함. v는 달라도 됨
- Transformer 그림으로 이해하기 [17:10]
    - Q, K, V를 계산하는 과정을 행렬 그림으로 표현 [18:10]
    - 메모리를 많이 먹는다(N^2의 시간복잡도) + Multi-headed attention(MHA) [19:00]
    - 트랜스포머는 인풋은 고정되어 있어도 주변 단어들에 따라서 인풋의 인코딩 값이 달라짐 (조금 더 flexible한 모델)
    - MHA의 장점 + 고려사항 [21:35]
    - Positional encoding이 필요한 이유 + 예 [23:50]
        - 데이터를 시퀀셜하게 넣었지만, 시퀀셜 정보가 포함되어 있지는 않음
        - self attention은 order에 independent하게 각 단어에 임베딩
    - Add & Normalize [25:40]
    - Decoder [26:40]
        - Decoder로 K, V를 보냄
        - 단어를 sequential 하게 생성
    - masking에 대해
        - i번째 단어를 만드는데 이미 모든 문장 알고 있으면 소용없음, 마스킹! (이전 단어들만 dependent, 뒤 단어들은 x)
- Vision Transformer [30:00]
    - 이미지분류 시, encoder 활용
- DALL-E [31:20]
    - Transformer의 decoder 활용, 이미지와 단어의 시퀀스를 조합
    - 문장에 대한 이미지를 만들어냄

# 3.8-실습 Transformer 실습

## Multi-Headed Attention

- 시작 [00:45]
    - import 및 각종 초기화
- Scaled Dot-Product Attention (SDPA) [01:35]
- Multi-Headed Attention (MHA) [07:35]
- 요약 및 강조 사항 [14:30]

## Position Embedding(강의에 안 나온 실습 코드)

- Check positional embedding
- Plot Sine Positional Embedding with different `embedding dimensions`
- Plot Sine Positional Embedding with different `steps`

# 기본 과제 5 Multi-head Attention Assignment

## Multi-Headed Attention(8강 MHA 실습과 동일)

- 시작
    - import 및 각종 초기화
- Scaled Dot-Product Attention (SDPA)
- Multi-Headed Attention (MHA)

# 3.9-1 Generative Models Part 1

## Introduction [0:30 ~]

- generative model: 단순한 생성형 모델?

### generative model의 학습[1:15 ~]

- Generation : 강아지를 학습했다면 새로운 강아지 이미지를 얻어낼 수 있어야 함.
- Density estimation : 어떤 사진이 주어졌을 때 강아지 같은지 판단.

### 기본 확률 분포 [3:00 ~]

- 랜덤 분포를 정의할 때, 나올 수 있는 경우의 수와 분포를 정의하는 파라미터의 숫자 정의하는 것이 중요!
- Bernoulli distribution : 동전을 던져서 앞/뒤인지 예측
    - 앞이 나올 확률 : p / 뒤가 나올 확률 : 1-p
    - X ~ Ber(p) | 라고 쓸 수 있음.
- Categorical distribution [4:00 ~] : m개의 면을 갖는 주사위
    - D = {1, 2, 3, … , m}
    - i가 나올 확률 = pi / sum(pi) = 1
    - Y ~ Cat(p1, p2, p3, …, cm) } 라고 쓸 수 있음.
- Example [5:15 ~]
    - 한 이미지의 한 픽셀이 가질 RGB의 경우의 수
    - 전체 경우의 수 = 256 x 256 x 256
    - 한 픽셀을 확률분포로 표현하기 위한 Parameter의 수 = 전체 경우의 수 -1

## Independence [6:10 ~]

- Example [6:40 : ~]
    - 흑/백의 픽셀을 갖는 binary image의 경우의 수?
    2 x 2 x 2 x … x 2 = 2^n
    - 필요한 Parameter의 수?
    2^n - 1
    - 모든 가능한 binary 경우의 수를 고려하는 것은 불가능

### Structure Through Independence [8:00 ~]

- Independence | P(X1, X2, X3, …, Xn) = P(X1)P(X2)P(X3)…P(Xn)
    - 필요한 경우의 수?
    2 x 2 x 2 x … x 2 = 2^n (서로 독립적이기 때문에 영향 X)
    - 필요한 Parameter 수?
    n개 (2^n - 1이 아닌 이유? →
- 독립이라는 가정을 하게 되면, 표현력이 굉장히 줄게 됨. 픽셀 별로 독립이라면 유의미한 학습을 시킬 수 있을까?

### Conditional Independence [11:00 ~ ]

- 모든 것이 joint distribute 된 것과 독립인 것 사이에 적합한 지점 찾기!

### 세 가지 규칙

- 연쇄 법칙(Chain Rule) [13:20 ~]
p(x1, x2, x3, …, xn) = p(x1)p(x2|x1)p(x3|x1,x2)…p(xn|x1,x2,x3,…,xn-1)
- Independence와 같은 규칙 없이 적용 가능

필요한 Parameter의 수 : 2^n - 1
(P(X1) = 1 / P(X2|X1) = 2(0 or 1) / P(X3|X1, X2) = 2^2 …

- 베이즈 룰(Bayes’ Rule)
    - 조건부확률 사이의 관계를 표현할 수 있음
- Conditional independence [15:30 ~ ]
z가 주어졌을 때 x, y가 Independent

필요한 Parameter의 수 : 2n - 1
(P(x1, x2, x3…xn) = P(x1)P(x2|x1)P(x3|x2)…(p(xn|xn-1) = 1+2+2+2+2+…+2)

## Autoregressive Models [17:50 ~]

- 순차적으로, 직전 몇 개 값에만 의존적인 모델
- 2~3차원 공간의 모델을 한줄로 피는 작업
- EX_ MNIST 숫자 데이터(28*28 size)
    - P(X) = P(X1, X2, …, X784)
    P(X1:784) = P(X1)P(X2|X1)P(X3|X2)…P(X784|X783)
- NADE(Neural Autoregressive Density Estimator) [20:10 ~]
    - Autoregressive를 가장 먼저 활용한 모델
    - Continuous random variable의 경우 mixture of Gaussian(MoG) 사용 가능

### Summary of Autoregressive Models [22:00 ~]

- 장점 : 샘플링이 쉽다.
단점 : 모든 데이터가 연속적으로 입력되어야 해서 생성이 느림. + 병렬화할 수 없음
- 장점 : explicit한 모델인 경우가 많다. (새로운 입력 X에 쉽게 대응)
확률을 계산하는데 빠름
- continuous variables로 확장하기 편하다

# DAY 13 회고

트랜스포머, BERT 등 오늘날 LLM의 기반이 되는 논문들을 여러 번 읽곤했다.

어렴풋이 안다고 생각해서 넘어갔는데, 팀원들이 조금만 다른 관점에서 질문하면 대답을 못했다.

학습 기시감을 해소하기 위해선, 여러 사람들과 자주 의견을 나눠보자.

공부할수록 어째 밑바닥만 드러나는 것 같지?? ㅎ

덧붙여, Transformer는 텍스트 데이터의 여러 챌린지를 이거 어때? 하고 휙 풀어낸 엄청난 모델이다. 아마 수 십년 후에 교과서에 실리지 않을까 할 정도…

겸손하게 공부를 이어나가자.

# 3.9-2 Generative Models Part 2

## Maximum Likelihood Learning [0:50 ~]

- 어떤 기준으로 좋음을 평가할 것인가?
(= 근사의 기준은 무엇인가?)
- KL-Divergence (이런게 있다 이해만 하기) [3:30 ~ ]
    - 두 번째 항을 최대화하는게 KL divergence를 최소화하는 것과 동일한 효과
    - MLE는 generative model을 풀 수 있는 쉬운 방법 중 하나
- Approximate the expected log-likelihood [7:00 ~]
    - Empirical Machine Learning: 모아진 데이터로만 기계학습 진행!
- Empirical Risk Minimization(ERM) [8:30 ~]
    - 한정된 데이터로 overfitting의 위험 있음.
    → hypothesis space를 줄임으로 overfitting 방지.
- Maximum Likelihood Learning은 under-fitting에 취약 [9:50 ~]
    - KL-divergence → Maximum
    - Jensen-Shannon divergence (수업에서 X, GAN에서 한번 등장)
    - Wasserstein distance (수업에서 X)
        - 새로운 확률분포의 evaluation metric을 정의할 때마다, 서로 다른 generative model 방법론이 나옴

## Latent Variable Models [11:45 ~]

- AutoEncoder는 generative model 아닌데, VAE는 맞음. 어떤 차이가 있을까?
- Variational Autoencoder [12:40 ~]
    - Variational inference(VI)
    너무 복잡한 분포를 간단한 분포로 변환 (근사시키고 싶음)
        - Posterior distribution → 너무 많으면 모델링할 수 없음(=손으로 풀 수 없음)
        - Variational distribution → 상대적으로 간단하지만 최적화 가능한 분포를 도입
        - VI를 활용하여 Posterior distribution 처럼 근사할 수 있음.
    - 수식 유도 [16:00 ~]
        - VAE의 핵심은 내가 처리하고 싶지만 건드릴 수 없는 VG를 가만두고, 건드릴 수 있는 ELBO를 최대화해서 Gap을 줄이고 싶어하는 것!
    - Evidence Lower Bound 수식 (ELBO) [19:00 ~]
        - 단점 : [20:45 ~]
        - intractable model
        → Maxinum Likelihood로 출발했지만 근사로 최적화하기 때문에 확률분포라고 보기 어려움.
        - 가우시안을 사용할 수밖에 없음.
        - Prior Fitting 항도 미분 가능해야하는데, KL divergence는 적분으로 이뤄짐. 적분 풀리기 위해서 isotropic Gaussian으로 가정해서 구함!
- 이미지 퀄리티만 봤을 땐, VAE 추천 x

## Generative Adversarial Networks(GAN) [23:25 ~]

- Discriminator: 내 이미지가 Generator가 학습한 것인지, 원래 데이터인지 구분하는 역할
- Generator: Discriminator를 속이는 역할
- G를 고정한 상태에서 D를 최적화하면 optimal point가 존재!
- GAN Objective [24:45 ~]
    - Jenson-Shannon Divergence(JSD)가 나옴. KL Divergence를 적당히 symmetric하게 만든 것(?)
    - discriminator(D) : G와 D의 밸런스 맞추는게 어려움

## Diffusion Models [28:55 ~]

- noise로부터 이미지를 생성 (GAN과 비슷함)
    - Diffusion 모델들은 이미지를 조금씩 변경시켜서 만듦. 하지만 성능이 엄청남!
- 천천히 변형시키기 때문에 효율적이지 않아 보이지만 성능은 뛰어남.
- Diffusion Process [31:40 ~]
    - Forward Process: Diffusion 모델은 이미지에 노이즈를 넣어서 이미지를 노이즈화 시키는 원리
    - reverse process : 노이즈를 없애고 이미지를 복원시키는 작업
    - (자세한 설명은 생략)
- 굉장히 오랜 스탭을 거쳐서 노이즈 벡터를 original 이미지로 refinement해나가는 과정
- DALL-E2 [34:05 ~]
    - Diffusion Model이 유명해진 계기.
    - DALL-E는 flip(?)이랑 diffusion이랑 섞어서 만듦
    - Image Editting이 가능하다는 특징 [35:10 ~]
        - 특정 영역에서만 변형하는 것 가능

# [Data Viz.] 3.1-1 Welcome to Visualization

- 데이터 시각화란? [1:20 ~]
- 강의 목표 [4:05 ~]
    - 시각화는 100점이 없음
    - 하지만 좋은 시각화는 만들 수 있음
- 목차 [5:30 ~]
- 강의 진행 방식 [7:50 ~]
    - 이론 후 실습

# [Data Viz.] 3.1-2 시각화의 요소 상태

## 데이터 이해하기

- 데이터 시각화 [0:55 ~]
    - 전체 데이터 - global
    - 개별 데이터 - local
- 데이터 셋의 종류 [1:40 ~]
    - (정형 / 시계열 / 지리 / 관계형(네트워크) / 계층적 / 비정형) 데이터
    - 정형데이터: 통계적 특성, 피쳐 간 관계
    - 대표적으로 4가지로 분류 [7:05 ~]
        - 수치형(numerical) 데이터 → 0이 존재
            - 연속형 (continuous) : 길이, 무게, 온도 등
            - 이산형 (discrete) : 주사위, 사람 수 등
        - 범주형(categorical) 데이터 → 0이 존재X
            - 명목형 (norminal) : 혈액형, 종교 등
            - 순서형 (ordinal) : 학년, 별점, 등급 등

## 시각화 이해하기 [9:30 ~]

- 마크(Mark)와 채널(Channel)
- Mark : 점, 선, 면으로 이루어진 시각화
- Channel : 각 마크를 변경할 수 있는 요소들
- 전주의적 속성 (Pre-attentive Attribute) [11:20 ~]
- 주의를 주지 않아도 인지하게 되는 요소
: 기울기, 길이, 넓이, 크기, 모양, 곡선, 색깔 등..
- 동시에 사용하면 인지하기 어려움
: 적절한 사용 필요.

# [Data Viz.] 3.1-3 Python과 Matplotlib

## Import Library

### pip install matplotlib or conda install matplotlib설치 후

> import matplotlib as mpl
import matplotlib.pyplot as plt
> 

## 기본 Plot [4:45 ~]

### Figure & axes

figure라는 큰 틀에 axes 서브 플롯을 추가
→ fig.set_facecolor(’black’) 코드를 실행해보면 figure 위에 subplot 2겹인 것 확인 가능

fig = plt.figure(figsize = (x, y))
ax1 = fig.add_subplot(1,2,1) # figure를 1행 2열로 나눈 후 1번째 칸에 위치
ax2 = fig.add_subplot(1,2,2) # figure를 1행 2열로 나눈 후 2번째 칸에 위치
plt.show()

## plt로 그래프 그리기 [11:50 ~]

### ax에 그래프 추가하기

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot([1,2,3]) # ax1에 순차적으로 적용되는 방법
ax2 = fig.add_subplot(212)
plt.plot([3,2,1]) # ax2에 순차적으로 적용되는 방법
plt.show()

## 객체지향 API사용하기 [14:00 ~]

- pyplot API → 순차적 방법
- 객체지향 API → 객체에 대해 직접 수정하는 방법 (강의에서 주로 사용)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot([1,2,3]) # ax1객체 지정하여 그래프 입력
ax2.plot([3,2,1]) # ax2객체 지정하여 그래프 입력
plt.show()

## Plot의 요소 알아보기 [16:00 ~]

### 한 subplot에 그래프 여러개 그리기

fig = plt.figure()
ax = fig.add_subplot()
ax.plot([1,1,1]) # 파란색
ax.plot([1,2,3]) # 주황색
ax.plot([3,3,3]) # 초록색, 같은 그래프의 경우 색깔이 달라짐

- 막대 - 선 조합이면 둘 다 파랑으로 시작
plt.show()

### 색상 지정하기 [18:25 ~]

- 보통 color 파라미터를 수정하고, c로 되어있는 것도 있음

fig = plt.figure()
ax = fig.add_subplot()
ax.plot([1,1,1], color = ‘r') # 빨강, ‘g’:초록, ‘y’:노랑, ‘w’:흰색, 한글자 색(투박하고 원색적)
ax.plot([2,2,2], color = ‘forestgreen) # CSS에서 사용하는 색깔 명칭(통용되는 색)
ax.plot([3,3,3], color = #000000) # hex code : 00 - R, 00 - G, 00 - B → black
plt.show()

### Text 사용하기 [20:40 ~]

- 범례(legend) 사용하기 <3~5, 14>
- title 사용하기 [ax title, fig title] <6~7>
- set/get → set : 설정하기, get : 받아오기 <6, 8~9>
- ticks/ticklabels → ticks : 축에 적히는 수 <8~9>
- text/annotate 표시 <10~13>

1 fig = plt.figure()
2 ax = fig.add_subplot()
3 ax.plot([1,1,1], label = ‘1’) # 1 라벨링
4 ax.plot([1,2,3], label = ‘2’) # 2 라벨링
5 ax.plot([3,3,3], label = ‘3’) # 3 라벨링

6 ax.set_title(’Basic Plot’) # ax subplot의 제목 설정하기, 상단에 위치
7 fig.suptitle(’fig’) # figure의 title도 따로 지정 가능

8 ax.set_ticks([0,1,2]) # x축에 0,1,2만 출력
9 ax.set_ticklabels([’zeor’, ‘one’, ‘two’]) # x축 0,1,2 이름을 변경

10 ax.text(x=1, y=2, s=’This is TEXT’) # (1,2) 위치에 s 출력
11 ax.annotate(text = ‘This is ANNOTATE’, xy = (1,2)) # xy 좌표에 text 출력

- annotate는 좌표 지정 느낌이라 해당 좌표에 화살표 표시도 가능
12 # ax.annotate(test = ‘arrow’, xy = (1,2), xytext = (1.2, 2.2),
13 # arrowprops = dict(facecolor = ‘black’)) → 해당 좌표를 가리키는 화살표 생성
14 ax.legend() # 이 코드 안쓰면 그래프에 안나옴, 위치 변경도 가능
15 plt.show()

# [Data Viz.] 3.2-1 Bar Plot 사용하기

## 기본 Barplot[ 01:35 ~ ]

- Barplot : 직사각형 막대를 사용해 데이터 값을 표현, **범주형 개별 비교, 그룹 비교에 적합, 가장 많이 사용**
- 수직 (vertical) : (default) `.bar()`
- 수평(horizontal) : 범주가 많을 때 적합 `.barh()`

## 다양한 Barplot [ 03:30 ~ ]

### Multiple Bar Plot [ 04:20 ~ ]

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

### Stacked Bar Plot [ 05:41 ~ ]

: 2개 이상의 그룹을 쌓아서 표현하는 Barplot → 그룹의 순서는 항상 유지

맨밑의 bar 분포는 파악하기 쉽지만, 그 외의 분포들은 파악하기 어렵다는 단점

- bar()에선 bottom parameter, bar()에선 left parameter 활용
- 이를 응용해서 전체에서 비율을 나타내는 percentage stacked bar chart도 존재

### **실습** [ 35: 30 ~ ]

- `bottom` 파라미터를 사용해서 아래 공간 비워 둘 수 있음
    
    ```python
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    group_cnt = student['race/ethnicity'].value_counts().sort_index()
    axes[0].bar(group_cnt.index, group_cnt, color='darkgray')
    axes[1].bar(group['male'].index, group['male'], color='royalblue')
    axes[1].bar(group['female'].index, group['female'], bottom=group['male'], color='tomato')
    
    for ax in axes:
    	ax.set_ylim(0, 350)
    plt.show()
    
    ```
    
- percentage stacked bar chart [ 38: 00 ~ ]

### Overlapped Bar Plot [ 09:04 ~ ]

- 같은 축을 사용해서 2개 그룹을 겹치게 해서 비교 but 3개 이상은 힘듦!
- 투명도를 조정하여 겹치는 부분 파악(alpha)

### **실습** [ 40:30 ~ ]

- 다양한 실험을 통해 투명도 조정하기 → 보통 0.7 많이 활용 ( 진한 색이면 0.5)

### Grouped Bar Plot [ 10:21 ~ ]

- 가장 추천하는 방법 → stacked barplot보다 분포를 더 잘 볼 수 있고, overlap 보다 가독성 good
- 그룹별 범주에 따른 bar를 이웃되게 배치하는 방법
- matplotlib에선 구현 까다로움. seaborn은 easy
- 그룹이 5개~7개 이하일때, 효과적 ( 그룹이 너무 많아지면 범주 내 그룹 비교가 돼서 bad..)
→ 그룹이 많다면 적은 그룹은 etc로 처리해서 따로 그려주기

### **실습** [ 41:30 ~ ]

- 3가지 테크닉으로 구현 가능
    - x축 조정
    - `width` 조정
    - `xticks`, `xticklabels`
        
        원래 x축이 0, 1, 2, 3로 시작한다면 - 한 그래프는 0-width/2, 1-width/2, 2-width/2 로 구성하면 되고 - 한 그래프는 0+width/2, 1+width/2, 2+width/2 로 구성하기
        
- 그룹이 N개일때? [44:43 ~ ]

## 더 정확한 Barplot [12:36 ~]

### Proportion Ink(잉크양 비례)

- 실제 값과 그에 표현되는 그래픽으로 표현되는 잉크 양은 비례 해야함!
- 항상 x축의 시작은 0 ( 사람들 인지에 방해 주지 않도록!)
- 막대 그래프에만 한정되는 것은 아니며, donut chart 등 다수의 시각화에서 적용
- 좀 더 예쁘다고 좋은 시각화는 아님

### **실습** [ 47:00 ~ ]

- 정확한 비교를 위해서는 축을 0으로

### 데이터 정렬

- 더 정확한 정보 전달을 위해 정렬은 필수
    - `sort_values()`, `sort_index()`
- 데이터의 종류에 따라
    - 시계열 : 시간순
    - 수치형 : 크기순
    - 순서형 : 범주 순서대로
    - 명목형 : 범주의 값에 따라 (최소 → 최대 or 최대 → 최소)
- 한 데이터에 대해 여러가지 기준으로 정렬하는 것이 좋은 인사이트 제공 → 대시보드에선 Interactive로!

### **실습** [ 48:44 ~ ]

### 적절한 공간 활용

- 여백과 공간만 조정해도 가독성 ⬆️
- matplotlib의 bar plot은 ax가 꽉 차서 답답
- X축 Y축 limit 줄이기 : `.set_xlim()`, `.set_ylim()`
- Spines : `.splines[spine].set_visible()` → 양변을 사각형처럼 안해줘도 ok - 트인 느낌
- Gap(width) → default 0.8 / 1이면 히스토그램 느낌 → 0.7 ~ 0.6으로 설정해서 가독성 ⬆️
- legend (범례) : `.legend()` → 범례 위치 조정해주기
- Margins(양 옆 테두리) : `.margins()`

### **실습** [ 49:30 ~ ]

### 복잡함과 단순함

- 필요없는 복잡함 NO
    - 무의미한 3D는 별로.. 최대한 지양 → 사람의 인지는 2차원에서 good (하고 싶다면 interactive로)
    - 직사각형이 아닌 다른 형태의 bar는 지양 ( 롤리팝 제외)
- 무엇이 보고 싶은가? (시각화를 보는 대상)
    - 정확한 차이 → grid good
    - 큰 틀에서 비교 및 추세 파악 → grid bad
- 축과 디테일 등의 복잡함
    - Grid(`.grid()`) : 큰틀에서 비교할 땐 최대한 grid 빼기
    - Ticklabels (`set.ticklabels()`)
    - Text 위치 (`.text()` or `.annotate()`)

### **실습** [ 51:54 ~ ]

- grid() 추가 → 어느정도 scale인지 알 수 있음

### 기타

- 오차 막대를 추가해서 Uncertainty 정보 추가 가능
- Bar 그래프의 변형 = Histogram (`.hist()` or barplot customize ) ! → Gap을 0으로 줘서 연속된 느낌을 줌
- 다양한 text 정보 활용하기
    - 제목 (`.set_title()`)
    - 라벨 (`.set_xlabel()`, .`set_ylabel()`)

### **실습** [ 53:04 ~ ]

- 오차막대를 사용하여 편차등의 정보 추가

## 코드 실습(기타)

- 각 barplot의 y축의 범위를 공유하기
    - 방법 1 : `sharey` 파라미터를 사용
    
    ```python
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True) # 같은 y축을 사용할 수 있음!
    
    ```
    
    - 방법 2 : 반복문을 이용해 y축 범위를 개별적으로 조정
    
    ```python
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].bar(group['male'].index, group['male'], color='royalblue')
    axes[1].bar(group['female'].index, group['female'], color='tomato')
    
    for ax in axes:
    	ax.set_ylim(0, 200) # y축의 범위를 0 ~ 200으로 지정
    
    plt.show()
    
    ```
    
    - 

# [Data Viz.] 3.2-2 Line Plot 사용하기

## 기본 Line Plot

### Line Plot이란? [ 01:00 ~ ]

- 연속적인 변화하는 값을 순서대로 점으로 나타내고, 이를 선으로 연결
- 왼쪽 → 오른쪽: 시간의 변화에 적합 → 추세를 살피기 위해 사용, 시계열 분석에 특화
- `.plot()` 사용

### 실습 [18: 10 ~ ]

- 정렬이 되어야 함
- 테크닉을 사용하면 정N각형이나 원도 그릴 수 있음

### Line Plot의 요소 [02:40 ~]

- 5개 이하의 선 사용을 추천 ( 그 이상은 가독성 떨어짐)
1. 색상 (color)
2. 마커 (marker, marker size)
3. 선의 종류 ( linestyle, linewidth) : 실선 기본, 점선은 보조정보나 비교군에 적용
`solid`, `dashed`, `dashdot`, `dotted`, `None`

### 실습 [21: 50 ~ ]

### Line Plot을 위한 전처리 [04:20 ~ ]

- 시계열 데이터는 Noise로 패턴 및 추세 파악 어려움

→ smoothing 활용 (ex. Moving Average 이동 평균) : 디테일은 안 보이더라도 trend 파악에 good

### 실습 [24: 45 ~ ]

이동평균 rolling method 사용

```python
google_rolling = google.rolling(window=20).mean()

```

## 정확한 Line Plot [06:00~]

### 추세에 집중

- 시계열 데이터는 추세에 민감 → 추세를 보기위한 목적 → 축을 0에 두기보다는 Max와 Min을 기준으로!
- Clean 하게 트렌드에 집중한 line plot (Grid, Annotate 생략해도 ok)

### 실습 [30: 30 ~ ]

### 간격 [08:10 ~]

- 규칙적인 간격 맞춰주기 : 기울기 정보로 변화량을 느끼기 때문에 x축의 간격이 상당히 중요, 없는 데이터가 있다고 오해 불러일으킬 수도!
- 규칙적인 간격의 데이터가 아니라면 각 관측값에 점으로 표시해주기

### 실습 [32: 32 ~ ]

### 보간[10: 30 ~]

- 점과 점 사이에 데이터가 없을 때, 이를 잇는 방법
- 데이터의 error나 noise가 포함되어 있는 경우, 데이터의 이해를 도움!
    - Moving Average
    - Smooth Curve with Scipy ( `scipy.interpolate.make_interp_spline()`, `scipy.interpolate.interp1d()`, `scipy.ndimage.gaussian_filter1d()` )
- 없는 데이터를 있다고 생각할 수 있기에 일반적인 분석에선 지양! (트렌드 보여줄땐 괜찮)

### 실습 [33: 15 ~ ]

### 이중축 사용 [12:10 ~]

- 한 plot에 대해 2개의 축을 사용. 서로 다른 종류의 데이터 표현도 가능
1. 서로 다른 데이터의 scale이 다를 때 : `.twinx()` 사용 - 반대쪽에 이중축
2. 한 데이터에 대해 다른 단위 : `.secondary_xaxis()`, `.secodnary_yaxis()` 사용

→ 같은 시간 축에 대해 서로 다른 두개의 그래프를 그릴 땐, 이중축 지양!

### 실습 [34: 08 ~ ]

### 기타[14:50 ~]

- 범례 대신 라인 끝 단에 레이블을 추가하는 것이 식별에 도움!
- Min / Max 정보는 추가해주면 도움이 될 수 있음 (annotation)
- 보다 연한 색을 사용해 uncertainty 표현 가능 (신뢰구간, 분산)

### 실습 [37: 30 ~ ]

# [Data Viz.] 3.2-3 Scatter Plot 사용하기

## 기본 Scatter Plot [01:00 ~]

### Scatter plot이란? [01 : 20 ~]

- 점을 사용하여 두 feature간의 관계를 알기 위해 사용하는 그래프
- `.scatter()`를 사용

### 실습 [14: 40 ~ ]

### Scatter Plot의 요소 [01: 50~ ]

- 다양한 variation 사용으로, 2차원 데이터 → N차원 데이터로 확장 가능
1. 색(color)
2. 모양(marker)
3. 크기(size)

### 실습 [16: 16 ~ ]

### Scatter Plot의 목적 [03:00 ~]

- 상관관계 확인 ( 양의 상관관계 / 음의 상관관계 / 없음)
- 군집 확인
- 값 사이의 차이 확인
- 이상치 확인

## 정확한 Scatter Plot [06:10 ~]

### 실습 [18: 40 ~ ]

### Overplotting[06:12 ~]

- 점이 많아질수록 점의 분포를 파악하기 힘들다
    - 투명도 조정
    - 지터링 (jittering) : 점의 위치를 약간씩 변경
    - 2차원 히스토그램 : 히트맵을 사용
    - Contour Plot : 분포를 등고선으로 표현
    - joint plot…

### 점의 요소와 인지 [08: 53~ ]

- 색 : 연속은 gradient, 이산은 개별 색상
- 마커( marker) : 점 개수 많아지면 구별 힘듦, 크기 고르지 x
- 크기 : 버블차트, 구별은 쉽지만 오용의 위험성, 관계보다는 각 점간 비율에 초점

### 상관관계와 인과관계[11: 20~]

- 인과관계와 상관관계는 다름 !!!!!

### 추세선 [12:26 ~ ]

- 추세선을 사용하면 scatter의 패턴 유추 가능
- 추세선 2개 이상은 가독성 떨어짐

### 기타[13:05~]

- Grid는 지양, 색은 무채색으로
- 범주형에선 heatmap과 bubble chart

# [Data Viz.] 3.3-1 Text 사용하기

## Matplotlib에서 Text [01:18 ~]

### Text in Viz [01: 27~]

- 시각화에서 줄 수 없는 많은 설명을 추가할 수도, 오해를 방지할 수도 있음!
- but, text를 과하게 사용하면, 이해 방해함

### Anatomy of a Figure [02:36 ~]

- Title : 가장 큰 주제 설명 `fig.suptitle()` , `ax.set_title()`
- Label : 축에 해당하는 데이터 정보 제공 `ax.set_xlabel()`, `ax.set_ylabel()`
- Tick Label : 축에 눈금을 사용하여 scale 정보 추가 → major, minor
- Legend(범례) : 2가지 이상의 데이터를 분류하기 위한 보조정보 `ax.legend()`
- Annotation(Text) : 시각화에 대한 설명 추가 `text()` , fig.text()

## Text Properties(실습) [05: 29 ~ ]

### Font Components [ 08:45 ~]

가장 쉽게 바꿀 수 있는 요소

- `family`
- `size` or `fontsize`
- `style` or `fontstyle`
- `weight` or `fontweight`

### Details [12:10 ~]

폰트 자체와는 조금 다르지만 커스텀할 수 있는 요소

- `color`
- `linespacing` : 줄과 줄 간의 간격
- `backgroundcolor` : 배경색(하이라이트)
- `alpha` : 투명도
- `zorder` : 배치 순서
- `visible` : text 보이는지 여부 ( True/ False)

### Alignment [13: 50 ~]

정렬과 관련한 요소들

- `ha` : horizontal alignment
- `va` : vertical alignment
- `rotation` : (default) horizontal
- `multialignment`

### Box[16:10 ~]

- `bbox` : 텍스트 테두리에 박스 추가, round와 round4 선호, 딕셔너리로 전달
    
    ```python
    ax.text(x=0.5, y=0.5, s='Text\\nis Important',
            fontsize=20,
            fontweight='bold',
            fontfamily='serif',
            color='black',
            linespacing=2,
            va='center', # top, bottom, center
            ha='center', # left, right, center
            rotation='horizontal', # vertical?
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4)
           ) # bbox를 이용해 back ground color 조정 가능
    
    ```
    
- [Drawing fancy boxes](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fmatplotlib.org%2Fstable%2Fgallery%2Fshapes_and_collections%2Ffancybox_demo.html)

### Text API 별 추가사용법 [18:15 ~]

- Title & Legend : 제목의 위치 조정, 범례 제목, 그림자 달기, 위치 조정 [19: 30 ~]
- Ticks & Text : tick을 없애거나 조정 , 막대 위에 text 추가 [23: 35 ~ ]
- Annotate : 화살표 사용하기(arrowprops) [28: 38 ~ ]

## 한글 in Matplotlib

- 한글 설정하기

# [Data Viz.] 3.3-2 Color 사용하기

## Color에 대한 이해 [ 00:48 ~]

### 색이 중요한 이유 [00:58 ~]

- 데이터를 구분하는 데 있어서 위치와 색이 가장 효과적
- 시각화에서 얼마나 매력적이냐도 중요 ( 색 조합, 깔끔)

### 화려함이 시각화의 전부는 아니다 [02 : 13~]

- 화려함은 일부분
- 가장 중요한 것은 독자에게 원하는 인사이트 전달

### 색이 가지는 의미 [03: 23~ ]

- 관습적인 색 정보 사용하기 ( 높은 온도 : 빨강, 낮은 온도 : 파랑)
- 기존 정보와 느낌 잘 활용하기
- 감이 없으면 다른 사례 스터디, 이미 사용하는 색은 이유 존재

## Color Palette의 종류 [04:50~]

### 범주형(Categorical) [05:05 ~ ]

- 독립된 색상으로 구성되어 범주형 변수에 사용
- 색의 차이로 구분, 최대 10개(그 외는 기타로)

### 실습 [ 17:50 ~]

- `from matplotlib.colors import ListedColormap` : 색들을 리스트로 전달하면 cmap으로 바꿔주는 함수

### 연속형(Sequential) [06: 52~]

- 정렬된 값을 가지는 순서형, 연속형 변수에 적합
- 연속적인 색상 사용
- 단일 색조로 표현하는 것이 좋고, 균일한 색상 변화 중요

### 발산형(Diverge) [08:10 ~]

- 연속형과 유사하지만 중앙을 기준으로 발산
- 상반된 색상 사용, 중앙의 색은 양쪽 점에서 편향되지 않도록!

## 그 외 색 Tips [09:30 ~]

### 강조, 그리고 대비 [09: 35 ~]

- 데이터에서 다름을 보이기 위해 Highlighting 가능
- 색상 대비 사용
    - 명도 대비 : 밝은색과 어두운 색 배치
    - 색상 대비 : 가까운 색은 차이 더 크게 (파랑보라, 빨강보라)
    - 채도 대비 : 채도의 차이 (회색주황) → 이걸로 하이라이트 활용!
    - 보색 대비 : 정반대 색상을 사용하면 더 선명 (빨강초록)

### 색각 이상 [11:50 ~]

- 삼원색 중 특정 색 감지 못하면 색맹
- 부분적 인지 이상이 있으면 색약
- 색 인지가 중요한 분야에선 이에 대한 고려 필수

## 실습 [13 : 10 ~]

- 색상 더 이해하기
    - 색을 이해하기 위해서는 rgb보다 hsl을 이해하는 것이 중요
    - **Hue(색조)** : 빨강, 파랑, 초록 등 색상으로 생각하는 부분
        - 빨강에서 보라색까지 있는 스펙트럼에서 0-360으로 표현
        - 색조의 차이가 클수록 빠르게 구별 가능
    - **Saturate(채도)** : 무채색과의 차이
        - 선명도라고 볼 수 있음 (선명하다와 탁하다.)
    - **Lightness(광도)** : 색상의 밝기

# [Data Viz.] 3.3-3 Facet 사용하기

## Facet

### Multiple View의 필요성

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

## Matplotlib에서 구현

### figure와 axes

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

### N x M subplots

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

### Grid Spec의 활용

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

### 내부에 그리기

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

### 외부에 그리기

- 여러 그룹을 보여주기 위해서는 technique 필요
1. 플롯을 여러개 그리는 방법
2. 한 개의 플롯을 동시에 나타내는 방법 → 비교에 가장 좋음
    1. 쌓아서 표현하는 방법
    2. 겹쳐서 표현하는 방법
    3. 이웃에 배치하는 방법

# [Data Viz.] 3.3-4 More Tips

## Grid 이해하기[00:40 ~]

### Default Grid [01:05 ~ ]

- 기본적인 grid는 축과 평행한 선을 사용해 거리 및 값 정보를 보조적으로 제공
- 기본적으로 무채색 사용 ( `color` )
- 항상 layer 맨 밑에 오도록( `zorder` )
- 큰 격자/ 세부격자 선택 가능 ( `which = ‘major’,’minor’, ‘both’` )
- x축, y축 따로, 동시에 사용가능 ( axis = `‘x’,’y’,’both’`)
    
    ### 실습 [09:23~]
    
    ax.grid(zorder=0, linestyle='--')
    

### 다양한 타입의 grid [02: 20 ~]

- 여러 형태의 grid 가 존재! → numpy + matplotlib으로 구현
    - 두 변수의 합 (x+y = c) : feature의 절대적 합이 중요한 경우 사용 (ex. 국어+수학)
    - 비율 (y=cx) : 가파를 수록 y/x가 커짐 (ex. 국어성적비율)
    - 두 변수의 곱
    - 특정 데이터를 중심으로 ( 동심원을 사용) : 특정 지점에서 거리를 살펴볼 수 있음

### 실습 [14:50 ~]

- 간단한 수식을 이용해서 설정

```python
# x+y = c
## Grid Part
x_start = np.linspace(0, 2.2, 12, endpoint=True)

for xs in x_start:
    ax.plot([xs, 0], [0, xs], linestyle='--', color='gray', alpha=0.5, linewidth=1)

# y = cx
radian = np.linspace(0, np.pi/2, 11, endpoint=True)

for rad in radian:
    ax.plot([0,2], [0, 2*np.tan(rad)], linestyle='--', color='gray', alpha=0.5, linewidth=1)

# 동심원
rs = np.linspace(0.1, 0.8, 8, endpoint=True)

for r in rs:
    xx = r*np.cos(np.linspace(0, 2*np.pi, 100))
    yy = r*np.sin(np.linspace(0, 2*np.pi, 100))
    ax.plot(xx+x[2], yy+y[2], linestyle='--', color='gray', alpha=0.5, linewidth=1)

    ax.text(x[2]+r*np.cos(np.pi/4), y[2]-r*np.sin(np.pi/4), f'{r:.1}', color='gray')

```

## 심플한 처리 [05: 45 ~ ]

### 선 추가하기 [05:50 ~]

- y축에 평행한 선, x축에 평행한 선, 평균 선 등 추가

### 실습 [22:10 ~]

- `axvline()`
- `axhline()`

### 면 추가하기 [06: 45 ~]

- x range, y range를 면적으로 표현 → 가독성 증가

### 실습 [25:15 ~]

- `axvspan`
- `axhspan`
- 특정 부분을 강조할 수도 있지만, 특정 부분의 주의를 없앨 수도 있음!

### 변 추가하기 [실습]

### 실습 [28:30 ~]

- `ax.spines` : 네 변을 조정
    - `set_visible` : 보여줄 지 말지
    - `set_linewidth` : 두께
    - `set_position` : 위치
- 중심 외에도 원하는 부분으로 옮길 수 있음
    - `'center'` -> `('axes', 0.5)`
    - `'zero'` -> `('data', 0.0)`

## Setting 바꾸기 [07:45 ~]

### Theme [07: 53 ~]

: 대표적으로 많이 사용하는 테마  `default`

- `fivethirtyeight` 이나 `ggplot` 많이 사용

### 실습 [33:00 ~]

- `mpl.rcParams` or `mpl.rc`
- `plt.rcParams.update(plt.rcParamsDefault)` : 이걸로 다시 돌릴 수 있음
- theme : rc를 전체로 바꾸고 싶다면!
    - `mpl.style.use('seaborn')`
    - 하나만 바꾸고 싶다면 with 문 이용
    
    ```python
    with plt.style.context('fivethirtyeight'):
        plt.plot(np.sin(np.linspace(0, 2 * np.pi)))
    plt.show()
    
    ```
    

# 프롬프트 웨비나

- Perplexity AI
- RAG 검색증강 생성
- GPTs
- Prompt Leak: SQL injection처럼 등록된 Instruction 질문하면 바로 알 수 있음
- 프롬프트 deepmind 'step by step'

# DAY 14 회고
벌써 한 주를 마무리하는 금요일이다. 내일은 아마 이것 저것 제출하느라 정신이 없을 것 같다.
의외로 이번주도 참 알차게 보낸 것 같다. 강의나 과제 퀄이 굉장히 좋은데 억지로 곱씹어볼 시간을 만들어야겠다.