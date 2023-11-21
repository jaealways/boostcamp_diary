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
    - 에타가 너무 크면 학습이 안됨
    - adaptive learning net: stepsize 자동으로 잡아줌

### Multi Dimensional [13:05~]

- 다차원에서 행렬 사용해서 아웃풋 구할 수 있음
- 행렬 연산이 어떻게 일어나는지에 대한 시각적 설명

### Beyond Linear Neural Networks [14:45~]

- 여러 개의 뉴럴넷을 층층히 쌓을 때, 비선형변환 필요
    - nonlinear transform 
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
    - entropy를 활용한 loss function: 뉴럴넷에서 해당하는 클래스의 값만 높이고자 함
    - 분류를 할 때, 다른 값들보다 높은 값만 도출하는 로직을 짜면 됨. 수학적으로 표현하기 어려워서 cross-entropy 컨셉 사용하는데 과연 이게 최선일까 고민해보기
- Probabilistic Task일 때
    - confident interval, uncertain 정보를 같이 찾고 싶을 때 사용


# 3.2-실습 Multi-Layer Perceptron

## 환경설정

### Dataset [01:05~]

- 필요한 라이브러리 import 설명
- MNIST 데이터셋 실행 및 간단한 설명

### Data Iterator [03:30~]

- Dataloader 및 batch size 실행 및 설명
    - shuffle=True로 해줘야 다음 epoch로 넘어갈 때 새롭게 섞어줌

## 모델 설명

### Define the MLP model [04:00~]

- Multi Layer Perceptron 사용
- 클래스 및 모듈에 대한 간단한 설명

### Simple Forward Path of the MLP Model [07:10~]

- Pytorch의 장점은 Session이 없다는 것, 바로 Network 만들고 Run 할 수 있음
- to(device) 사용하기
- MLP에서 인풋, hidden, 아웃풋까지의 간단한 메커니즘 설명
    - forward 명시 안해줘도 동작함

### Check Parameters [09:15~]

- 파라미터를 shape와 연관시켜서 설명
    - Dense layer에 필요한 파라미터가 convolution보다 훨씬 많음??

### Evaluation Function [10:10~]

- Validation Accuracy 측정방법 설명

### Initial Evaluation [11:00~]

## 모델 학습

### Train [11:25~]

- 파라미터 초기화하고 배치별로 루프
- training accuracy 개선되는 것 확인

### Test [13:50~]

- 25개 랜덤샘플이 얼마나 정확도 보이는지 체크
- Model Initialize하고 측정하면 낮은 정확도 보임



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
    - Validation 데이터의 error 값이 커지기 시작할 때 스탑
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



# 3.3-실습 Optimization Methods
- 어떻게 서로 다른 optimizer가 영향을 미치는가
- install & import [01:20]
- dataset [02:47]
- model [03:20]
- models [04:20]
- train [05:50]
- result [07:20]
    - adma: 모멘텀 + adaptive learning 합친 것, adam이 잘되었기 때문에 둘다 쓰는게 중요
    - adam은 어느 파라미터에선 learning rate를 줄이고 어디서 늘리는 것을 왔다갔다? 학습속도가 훨씬 빠름?
    - SGD와 모멘텀은 왜 차이가 날까?: 모멘텀은 이전 gradient 사용해서 다음번에도 사용하겠다는 의미, 이 둘의 차이
    - SGD는 가장 피크가 큰 지점 위주로 맞추고 있는데, 제곱에러를 사용하기 때문에 거리차이가 큰 지점 먼저 수렴함. 이상치에 민감
- 일단 처음엔 Adam 쓰는 것 추천!!


# 기본 과제 2 Optimization Assignment

## **Regression with Different Optimizers**

- dataset
- model
- models
- check parameters
- train
- result



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
- **fully connected layer** : decision making  (→ parameter 개수를 줄이기 위해 layer 감소)
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
    - 채널 차원 감소를 위함 → #parameter 감소 & #layer 증가
    - ex. Resnet, Densenet
    - bottleneck 아키텍처
        - 앞뒤로 1by1 넣으면 파라미터 줄일 수 있다...
    - cnn에서 bias 고려하면 어디항에 +1?
        - 각 필터에 대한 파라미터 수는 (3*3*64=576), (576+1)*32

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

### 6. evaluation  (이전 실습과 동일) [11:03]

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


# 기본 과제 3 CNN Assignment

## **Convolutional Neural Network (CNN) - MNIST dataset**

- import
- dataset
- data loader
- model
- check parameters
- forward
- evaluation
- train
- test

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