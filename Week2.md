# Week2: Pytorch Basics for AI

## Day 6 (2023-11-13)

### Pytorch

- 딥러닝 프레임워크가 구현되는 것 배우고 싶으면 "밑바닥부터 시작하는 딥러닝" 추천
- 표준화된 오픈소스는 굉장히 편함. 자료도 많고, 관리도 잘 되어 있음
- 현재는 pytorch, tensorflow 둘 이 표준으로 사용됨
- Keras 쉬움, 현재 tensorflow에 병합
- Tensorflow EAGER EXECUTION: 굉장히 반발
- 텐서플로우는 Production, 클라우드 사용, Multi-GPU 등 장점
- 파이토치는 디버깅이나 아이디어 쉽게 구현하는데 강점
    - Numpy + AutoGrad + Function
- python 기반의 수학적 연산 다루는 것은 대부분 numpy 기반
- 넘파이 ndarry, pytorch는 tensor
- 파이토치 대부분 사용법이 numpy랑 동일
- reshape 대신 view 사용, view랑 reshape 메모리 보장해줌???
- 파이토치 내적 구할 땐 dot, 행렬연산에선 mm(matmul) 사용
- mm은 broadcasting 안됨, matmul은 됨

```python
a=torch.rand(5,2,3)
b=torch.rand(5)
a.mm(b)
=> 연산 안됨
a.matmul(b)
=> 연산 됨

```

- torch nnFunction
- backward (자동 미분 지원)
- 주피터로 초반에 할 수 있지만, 깊게 들어가면 한계가 있음
[Pytorch 템플릿](https://github.com/victoresque/pytorch-template)
- 주피터에서 git clone 써서 데이터 가져오기, 매직메서드 !
- 외부에서 쉽게 ssh로 접근할 수 있게 만들기: ngrok, colab-ssh (vscode에서 remote-ssh 설치)
- __getattribute: 환경설정 json 바꾸고, 추가설정만 바꿈?
- train, trainer 초기설정 잡아주는 파일, 실행하는 파일?
- 주석을 달아가면서 이해?? 클론코딩 경험 있는 친구?


## Day 7 (2023-11-14)

### Linear Algebra
- 발표 준비를 하면서 선형대수 리뷰를 다시 해보는데, 생각을 조금만 비틀면 모르는 것 투성
- 두 벡터 간의 독립을 따질 때, 스칼라곱만 적용가능한가? 그렇다면 차원이 하나 커져서 행렬이 된다면?
- 통계학에서 말하는 독립과 선형대수의 독립이 온전히 같은 의미인가? Orthogonal하면 선형대수에서 독립이니까 통계적으로도 독립인가?(독립보단 uncorrelated하다는 표현이 좀 더 적합해보임)

### 기본과제1
-  [ 퀴즈 ] PyTorch Release Status (배포 상태)
    - Backwards compatibility가 무엇인가?
    - binary distributions like PyPI or Conda?
- zeros, zeros_like 차이
- gather 함수 정확히 이해하기
- torch.arange(11).chunk(6) vs torch.chunk(arange(11),6)



## Day 8 (2023-11-15)


### pytorch
- optimizer(자동미분) 어떤 과정으로 일어나는가?
- 딥러닝 아키텍처는 블록 반복의 연속 (nn.Module)
    - Input
    - Output
    - Forward
    - Backward
- nn.Module에서 attribute가 될 때는 required_grad=True (AutoGrad)
- 파라미터를 기본적으로 직접 설정하는 일은 거의 없음
- tensor로 할지, parameter(미분 일어나지 않으면 print 안됨?)로 할지?
- 파라미터를 직접 부를 일은 거의 없음. 구조에 대해 이해하기 위해 parameter, tensor 이해해야
- Data Centric AI??
- 모델에 데이터를 먹이는 방법
- class Dataset 
    - __init__ 초기 데이터 생성 방법 지정(디렉토리 설정 등)
    - __getitem__ (map-style): 하나의 데이터를 불러올 때 어떻게 반환해주는지
- class transforms: 이미지 등 데이터 전처리
    - __call__ ??
- DataLoader: batch, shuffle
- image의 tensor 변화는 학습에 필요한 시점에 변환? CPU와 GPU 사용과 연관되는 것 같음
- DataLoader: sampler, batch_sampler, collate_fn(Data Label 쌍을 Data 쌍 Label 쌍으로 변환)
- ** torchvision datasets 클론코딩 권장 **
- tqdm 사용해서 어느정도 진행되는지 확인하기!!!
- pytorch 기본 코드를 보면서 이해를 하는게 좋음(헷갈리면 Objected Oriented의 개념이 약해서 그럴 가능성 높음)
- next(iter(DataLoader))가 정확히 하는 역할??
- Pytorch에선 __init__, __getitem__, __len__ 매직 메서드를 필수적으로 오버라이딩해야 함

### Transfer Learning

- AI 가장 기본 되는 모델: backborn?, 최근은 fine-tuning이 대세
- 학습 한 번 하는데 공수가 크게 들기 때문에, 중간에 학습 결과 잘 저장하는게 중요 model.save()
- 모델 형태와 파라미터를 저장할 수 있음
- state_dict: 모델의 파라미터를 표시 및 저장(확장자 .pt, pth는 사용 안하는게 좋음)
- google drive에서 파일 옮기는 거 추천
- checkpoints: 학습 중간 결과를 저장하여 최선의 결과를 선택
- BCEWithLogitsLoss

- 다른 데이터셋으로 만든 모델을 현재 데이터에 적용
- 보통 TorchVision 사용, NLP는 HuggingFace가 표준
- Freezing: 특정 위치까지 멈춘다음 파라미터 위치 안바뀌게 함. (일부 레이어에서만 backprop 가능하게)
- Frozen 중간에 시키면 반드시 전에 돌렸던 코드 같이 실행시키기?

### Monitoring Tool
- tensorboard, weigth&biases 도구
- tensorboard
    - scalar
    - graph
    - histogram
- wandb
    - config: 하이퍼파라미터 정보를 넣어줌

### 기본과제1
- 그래서 identity가 뭐하는거?
- super().__init__()?? [참고자료](https://stackoverflow.com/questions/63058355/why-is-the-super-constructor-necessary-in-pytorch-custom-modules)
- BLAS vs LAPACK


### 두런두런: 변성윤 마스터님
- OCEAN 검사

## Day 9 (2023-11-16)

- 과제 1 부덕이 모델 수정하기 - module 참조 제거 (해결하기)
- 과제 1 [ 코딩 ] BatchNorm1d 분석해보기
- 과제1 forward hook, backward hook, gradient
- 5강 퀴즈 마지막 문제 다시 보기!
- 강의에서 진행해주신 ipynd 파일 모으기 (점심)

### Multi-GPU

- Multi-GPU를 어떻게 다룰까?
- to() GPU 교차
- pin_memory: 데이터를 간소화해서 바로 올리는거???
- num_workers: GPU x 4
- DataParallel의 문제 중, 한 GPU가 병목이 될 때 다른 GPU도 영향?
- Model 병목, 데이터 분산 불균형, GIL, Batch 사이즈 감소, 프로세스 분산 불균형


### Hyperparameter Tuning

- 모델, 데이터, 하이퍼파라미터 튜닝 중 데이터가 가장 중요(가장 성능 개선에 큰 영향 미침)
- NLP는 트랜스포머, CV는 RESNET, CNN 등이 일반화되어 있음. 모델을 바꿈으로 인해 생길 수 있는 유익에 한계
- NAS, AutoML
- 하이퍼파라미터는 예전만큼 중요성이 낮아졌음(가성비가 낮아짐)
- Grid Layout vs Random Layout

### PyTorch Troubleshooting

- OOM: Out of Memory
- Batch Size -> GPU clean -> Run
- CNN 직접 쌓음: Memory Buffer
- cuda.empty_cache()??
- tensor로 처리된 변수는 

### 마스터클래스

- AI가 한 것과 사람이 한 것을 구분 못하는 시대가 왔다
- 수능 상위 5%
- Stability(이미지) vs ChatGPT(텍스트)
- 아직도 pytorch?
- 정해진 task는 AI 이길 수 없음
- 논문 구현 잘하면 AI 잘하는 사람 -> 패러다임 바뀜(모델 구현하는게 AI 연구자의 메인 아닐수도)
- 언어모델: 언어생성모델(?)
- 하루 서버 비용만 수 백억, 이미 돈 놀이 느낌?
- 하이닉스: HBM 만드는데, Nvidia 부상하면서 같이 상승
- 앞으로 치열한 전쟁터는 하드웨어(?)
- 이제는 AI는 우선 ChatGPT 쓰고 데이터 모아서 AI 대체하는 트렌드

- LLM이나 프롬프트 러닝 못하면 의미 없을수도
- LLM으로 DX하지 않는 조직은 살아남지 못할 것
- AI가 생성하고, 인간이 검수하는 시대
- 조금이라도 빨리 회사에 들어가야 함

- GPTs 열심히 하기!!
- Data Centric AI가 무슨 의미?
- 7B 짜리 모델과 ChatGPT API 정도는 경험해보기
- 앞으로는 ChatGPT API 사용해서 필요한 것 만드는게 중요

- 어떤 문제를 풀면서 어떤 느낌을 받았냐? 나만의 edge?
- ex) 백엔드: 시간당 10만 개의 리퀘스트 처리
- 두 세줄로 짧게 직관적으로 설명하기
- Front or Back + GPT로 퍼블리시
- 학원다니듯이 공부하는건 의미 없음. ChatGPT + 유튜브 잘 사용하기
- 


## Day 10 (2023-11-17)



## 개념정리


