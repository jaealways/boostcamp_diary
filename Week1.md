# Week1: AI Math & Python Basics for AI

## Day 1 (2023-11-06)

### LLM의 방향성 (성킴 교수님)

- 최근 OpenAI는 모델 숨기고 있음. LLAMA는 모델을 공개함. 이를 기반으로 많은 시도들 일어남
- 프랑스 스타트업 Mistral AI가 굉장히 최근 높은 성능 보임
- 파인튜닝이나 프롬프트 러닝을 통해 무엇인가를 하려는 니즈도 많지만, 모델에 대한 깊이 있는 이해가 필요함


### 피어 투 피어

- 피어 세션은 허울 뿐인 것이 아니라 오랜 과정을 거쳐 효율성이 입증됨


### Python 기초

- 두 변수 메모리 참조 말고 복붙하고 싶으면,
```
a=[1,2,3]
b=[x for x in a]
b=a[:]
```
- 두번 째 b도 가능. 단 2d array에선 작동 안함. Deepcopy 써서 해야
- formating % 사용시 10%.2f 등 (총 10자리로 나오는데 소숫점은 두자리까지 표시)
- <, >, ^(가운데) padding(정렬) 0<10s 등
- is는 memory의 주소 비교, ==는 값 비교
- python에선 -5~256까지 메모리 저장. 그 이외 숫자 넘으면 새로운 메모리 만들기 때문에 is값 달라짐(is 사용 지양할 것)
- 여러 boolean 따질 때 [True, False, False] 같이 list처럼 all([True, False, False]), any로 사용
- 삼항연산자 깊이 공부할 것
- 각 type들이 메모리 몇 byte 차지하는가?? -> CS 기본 개념인데 부족한 듯 하다...
- 문자열 앞에 print(r"~~ \n") 작성하면, \n 그대로 출력됨
- String 대소문자 구분, 컴퓨터는 언제 둘을 같게, 다르게 인식하나?
- Call by Value, Call by Reference 잘 구분하기(코테 재귀 사용하면 자주 등장)
- 코딩 컨벤션 잘 따르기. 사람이 이해하기 쉬운 코드를 만들어야한다!!
- memory address, iterable, next()
- asterik(*) 가변인자
```
def asterisk_test(a,b, *args):
    print(list(args))

asterisk_test(1,2,3,4,5)

[3,4,5]

```

- keyword 가변인자: **kwargs (파라미터 여러개 넣을 수 있음)


### 멘토링

- 최근 트렌드를 보면 데이터에 진심인 사람이 더 중요하다
- 석박사들은 벤치마크 데이터셋에 매몰되는 경향이 있음. 오히려 데이터 품질 높이고 데이터셋을 잘 정제하는 것이 더 중요할 수 있음
- 리눅스 잘 다루자!!!



## Day 2 (2023-11-07)

### Python 기초

- __init__(속성정보), __main__, __add__ 등 다양한 네임 맹글링 숙지하기
- 객체, 함수는 결국 복잡한 로직으로 가면 잘 활용하는게 중요
- Visibility 할 때, 어떤 기준으로 캡슐화 또는 정보 은닉해야하나?
- __item : two underbar Private 변수로 외부에서 접근 못함
- Decorator 의 종류와 사용 목적에 대해 잘 생각해보기
- First-class object: 함수 자체를 변수로 할당해서 다른 함수의 파라미터로 사용할 수 있음 가령

```python

def square(x):
    return x*x

def cube(x):
    return x*x*x

def formula(method, argument_list):
    return [method(value) for value in argument_list]    

```

- argument 리스트에 별도로 상황에 맞는 함수 지정해서, loop마다 처리하게 할 수 있음(번거롭게 데이터 분할하지 않아도 됨)
- Inner Function(함수 속의 함수)의 변수는 어디까지 영향 미칠까??
- 데코레이터에 함수 할당 가능??

```python
def star(func):
    def inner(*args,**kwargs):
        print("*" *30)
        func(*args,**kwargs)
        print("*" *30)
    return inner

@star
def printer(msg):
    print(msg)
printer("Hello")

```

- 데코레이더의 정확한 의미가 뭘까?? 고민해보고 찾아보기
- 라이브러리, 모듈, 패키지 개념적 차이
- __pycache__: 메모리 로딩위해 컴파일 시킴. 좀 더 찾아보기
- pip는 컴파일이 잘 안되어 있음, conda 가상환경 자주 사용
- 전체를 Exception 잡으면 어디서 에러가 발생했는지 알기 힘듦
- try except else, finally 등 사용 가능함
- **사용자가 입력해서 에러가 발생해야한다면 try except, 정상적으로 처리되야 한다면 if else 사용 권장**
- assert(~): ~가 False면 error 발생
- Text(보통 메모장으로 열리면)와 Binary(xls, hwp 등) 파일 잘 구분하기
- with open: with 안에서만 파일 열림
- utf-8로 데이터 맞추는게 호환성 좋음
- 윈도우, 맥, 리눅스는 디렉토리 간 separator 다르기 때문에 '\' 사용하지 말고, join + shutil.copy 사용 권장(또는 pathlib)!
- 객체는 원래 메모리에 있어야 하는데, Pickle은 객체를 영속화하는 것! (클래스도 Pickle로 저장 가능)
- Logging DEBUG INFGO WARNING ERROR CRITICAL 등 종류 숙지
- argparser: argument 입력으로 실험 가능하게?
- 정규표현식 사용해서 효율성 높이자! *,^ 등 기호 숙지하기

### numpy

- array shape 차원에 대한 이해 (numpy 차원 축소 합칠 때 헷갈리지 않게), 특히 numpy 쓸 때 항상 차원 이슈 많았음
- nbytes 별 용량 잘 기억하기
- array[:,::2] 2개씩 건너뛰기 start:end:step
- empty, 메모리와 가비지 콜렉션의 관계?
- numpy eye 단위행렬에서 k 쓰면 어떻게 나오는지 잘 숙지하기 (k가 음수면?)
- 차원 증가하면 axis가 어떻게 변화하는지 잘 숙지하기
- array+scalar 또는 array+matrix 등 shape가 다른 연산을 broadcasting이라 함. 어떻게 데이터가 변환되는지 잘 숙지하기
- logical_and, logical_or
- np.where 1) True, False에 해당하는 값 넣거나, 2) Index 값 반환하도록 사용
- argsort, argmin, argmax idx값 뽑아주는 메소드


### pandas

- 타입 변경: astype() (numpy에도 있었음)
- 보통은 df를 객체로 부르지 않고, csv를 많이 부름
- Series만들 때, index=[0,1,3]으로 지정하면 3 size가 아니라 4 사이즈(idx 2가 비어있는) 모양으로 생성됨
- del은 df에서 아예 삭제, drop은 메모리에서 임시 삭제(다시 실행하면 원본 나옴)
```python
df['account']
df[['account']]
```
- 첫번째는 series, 두번째는 df로 반환
- inplace 써서 삭제, 쓸 때와 안 쓸 때의 차이?
- replace 자체로는 원본 안바뀜. 바꾸려면 inplace 써야
- apply, map, lambda 등 고급문법 복습
- group by, join 등 sql과 유사

```python
obj = pd.Series([1,2,3], [‘a’,’b’,’c’], dtype=np.float32, name=”example”)
```
- 두 번째 array는 인덱스
- loc, iloc 헷갈리지 말기


## Day 3 (2023-11-08)

### AI MATH

- 내적 정의 복습하기
- trace, rank 등 선형대수 리뷰
- numpy의 inner는 선형대수 내적과는 다름
- 행렬곱을 통해 다른 차원 공간으로 보낸다는 의미로 이해할 수 있음
- 역행렬 연산이 불가능할 경우, 유사역행렬 사용(수학적 background에 대해 찾아보기), numpy에선 pinv로 가능
- Moore-Penrose 역행렬
- sin, cos 미분 복습


### Gradient Descent

- 역행렬을 사용하지 않고도 최적점을 찾을 수 있음
- 확률적 경사하강법: 극소점에서 탈출 가능함, 하드웨어, 알고리즘 효율성 등 고려했을 때 필요함


### Neural Network

- softmax시 너무 큰 값 들어오면 overflow 발생. Overflow 정확한 뜻?
- 층이 깊어질수록 적은 뉴런의 수로 근사시킬 수 있음
- 자동미분
- 연쇄법칙에 대한 좀 더 직관적인 이해
- (추가) FeedForward Network는 가중치를 어떻게 변경할까?


## Day 4 (2023-11-09)

- FRM 시험으로 결석


## Day 5 (2023-11-10)

### Statistics

- 주변확률분포, 결합분포 개념 공부하기
- 회귀문제는 조건부확률이 아니라 조건부기대값으로 추정. Why??
- 연속이냐, 이산이냐에 따라 적분이냐 급수냐 취하는 방법이 다름
- 확률분포를 명시적으로 알 수 없을 때, 몬테카를로 샘플링을 통해 진행
- 모수적 방법론, 비모수 방법론 (모수가 없는게 아니라, 데이터에 따라 분포가 바뀌거나 모수가 무한히 많은 경우일수도)
- 표본분산 구할 때 N-1로 나누는게 불편추정량
- 중심극한정리: 표본이 정규분포를 따르는게 아니라, 표본평균이 정규분포를 따름
- 로그가능도 사용하면 곱셈이 아니라 덧셈으로 치환가능. 데이터가 굉장히 많아지면, 연산오차 발생. 시간복잡도를 O(n2)에서 O(n)으로 줄일 수 있음
- 카테고리분포에서 Multinoulli를 따르는 변수 P1,...pd를 따르는 분포는 모두 합해짐
- 라그랑주를 통해 새로운 목적식을 만들 수 있음
- 딥러닝 최대가능도 추정법 잘 이해하기
- KL 다이버전스, 와셔슈타인 거리 등
- 두 개의 확률분포 사이 거리를 최소화하는 것 = 목적으로 하는 확률분포의 최적화된 모수를 구하는 것

### Bayes Theroem

- 베이즈 정리 이해하기 위해선 조건부확률 이해해야 함
- 사전확률: 데이터가 주어지지 않았을 때 얻을 수 있는 분포
- 사후확률: Evidence를 통해 업데이트할 수 있는 확률 분포
- 얻을 수 있는 통찰: Evidence(새로운 데이터)를 지속적으로 업데이트해서 새로운 분포를 추정할 수 있다
- True Positive, False Negative를 골고루 볼 수 있을 때 모델의 정확도에 대해 정밀한 측정을 할 수 있음
- 의료에선 False Negative가 중요한 이슈
- ** 조건부확률을 통해 함부로 인과관계를 추론하면 안됨 **
- 인과관계를 알기 위해선 중첩요인효과를 제거해야 함
- 키가 크면 지능지수가 높다?? (키가 큰 어른이 지능검사가 높을 가능성 높음). 인과관게 추론 공부하기
- 도메인 지식을 활용해서 인과관계 정확히 파악해야 정밀한 모델 만들 수 있음


### CNN

- 커널을 통해 국소적으로 증폭 감소시킴
- 엄밀히 말하면 cross-correlation 연산임
- elementwise 곱을 통해 아웃풋
- 3차원 이상일 경우, 각 채널 별로 convolution 연산을 수행하는 것이라 이해할 수 있음
- 역전파를 사용해도 똑같이 convolution 연산이 나오게 됨 (Discrete일 때도 이해하기)
- 컨볼루션 역전파 이해하기!!

### RNN

- 


### 오피스아워

- 언더바 _ 사용시 메모리 줄일 수 있음
