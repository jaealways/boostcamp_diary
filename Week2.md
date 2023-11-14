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

### 과제1
-  [ 퀴즈 ] PyTorch Release Status (배포 상태)
    - Backwards compatibility가 무엇인가?
    - binary distributions like PyPI or Conda?




## Day 8 (2023-11-15)

## Day 9 (2023-11-16)

## Day 10 (2023-11-17)

