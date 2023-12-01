- 4일 동안 예비군 참석으로, 꼼꼼히 학습하지 못함
- 팀에서 만든 목차로 대신

# Data Visualization(2)

# (4-1강) Seaborn 소개

## Seaborn 소개 [00:40]

- matplotlib 기반 통계 시각화 라이브러리
- 방법론 위주로 설명 예정 [01:20]

## 다양한 API [03:10]

- Categorical API
- Distribution API
- Relational API
- Regression API
- Multiples API
- Theme API

# (4-2강) Seaborn 기초

## 1. Seaborn의 구조 살펴보기 [01:00]

- 1-1. 라이브러리와 데이터셋 호출
- 1-2. Countplot으로 살펴보는 공통 파라미터

## **2. Categorical API [09:15]**

- 2-1. Box Plot
- 2-2. Violin Plot
- 2-3. ETC

## **3. Distribution [24:35]**

- 3-1. Univariate Distribution
- 3-2. Bivariate Distribution

## **4. Relation & Regression [38:00]**

- 4-1. Scatter Plot
- 4-2. Line Plot
- 4-3. Regplot

## **5. Matrix Plots [43:30]**

- 5-1. Heatmap

# (4-3강) Seaborn 심화

## **1. Joint Plot [01:20]**

- hue를 사용하여 구분
- 다양한 종류로 분포 확인

## **2. Pair Plot [04:35]**

- pair-wise 관계 시각화
- 2가지 변수를 통해 조정(kind, diag_kind)

## **3. Facet Grid 사용하기 [08:45]**

- 3-1. catplot
- 3-2. displot
- 3-3. relplot
- 3-4. lmplot

# (5-1강) Polar Coordinate

- PolarPlot [0:35 ~]
    - 회전, 주기성 등을 표현하기에 적합
    - Scatter, Line, Bar 모두 가능
    직교 좌표계 X, Y에서 변환 가능
    - X = R * cos x
    - Y = R * sin x
- Radar Plot(= Star Plot) [3:00 ~]
    - 오각형, 육각형으로 능력치 표현할 때 자주 사용
    - 사용 주의점
        - 각 feature는 독립적이며 척도가 같아야 함
        - feature 순서에 따라 값이 달라보임(면적의 차이)
        - feature가 너무 많으면 가독성 떨어짐
- 실습 [7:10 ~]
    - Polar Coordinate [8:05 ~]
    1. Polar Coordinate 만들기
    2. Polar Coordinate 조정하기
       [set_rmin - 반지름 시작점 조정
        set_rmax - 반지름 최대값 조정
        set_rticks - 반지름 표기 지정
        set_rlabel_position - 반지름 label이 적히는 각도 조정
        set_thetamin - 각도의 최소값
        set_thetamax - 각도의 최대값]
    3. Polar 기본 차트 [13:40 ~]
       [Scatter() - 기존 산점도와 같은(theta, r 순서)
        Bar() - 막대의 모양이 기존과 갈라서 크기비교 어려움
        Line() - 나선형 모양이 될 수 있음
        fill()  - 선 아래 부분을 채워 면적을 살펴볼 수 있음
    - Radar Cart [16:15 ~]
    1. Radar Cart 기본 틀 구성
        테두리 그릴 때 마지막 선이 안채워져서 첫 값을 마지막에 추가
    2. 커스텀 및 조정 [19:25 ~]
        set_thetagrids - 각도에 따른 그리드 및 ticklabels 변경
        set_theta_offset - 시작 각도 조정
        여러개의 데이터를 비교할 때 더 효과적

# (5-2강) Pie Charts

- Pie Chart [0:40 ~]
    - 원을 부채꼴로 분할하여 표현
    백분위로 나타낼 때 유용
    - 가장 많이 사용하는 차트지만 사용 지양
    - 비교가 어려움
    - 유용성 떨어짐
    - 오히려 bar plot이 유용(각도보다 길이가 더 효과적)
- Pie Chart 응용 [2:05 ~]
    - Dount Chart : 중간이 비어있는 Pie Chart
    - Sunburst Chart : 햇살을 닮은 차트
- 실습 [4:20 ~]
    - Pie Chart [4:40 ~]
    1. Pie차트와 막대 블럭
        Pie 차트는 비육정보를 제공.
        하지만 구체적인 양의 비교가 어려움
        막대블럭은 양의 비교를 쉽게 할 수 있음
    
    2. Pie Chart Custom [7:55 ~]
        startangle - 90을 넣으면 가장 위에서 부터 시작
        explode - 한 조각을 강조
        shadow - 3D느낌으로 보여줌(가독성은 떨어져도 집중시킴)
        autopct - 차트의 비율 자동 계산
        labeldistance - label 길이가 길어지면 차트와 떨어뜨림
        rotatelabels - 중심점을 기준으로 차트와 같은 방향으로 돌림
        counterclock - 시계방향으로 그래프 생성
        radius - 차트의 반지름(크기) 조정
    - Dount Chart [12:20 ~]
    중앙에 흰 원(background 색과 같은 색)을 그려줌
        pctdistance - 중심에서 떨어진 거리
        textprops - 비율 글씨

# (5-3강) 다양한 시각화 라이브러리

- Missingno [1:10 ~]
    - 결측치를 체크하는 시각화 라이브러리
- Treemap [2:35 ~]
    - 계층적 데이터를 직사각형을 사용하여 포함관계를 표현
    - 사각형을 분할하는 타일링 알고리즘에 따라 형태가 다양해짐
- Waffle Chart [5:20 ~]
    - 와플 형태로 discrete하게 값을 나타내는 차트
    ex_ 공연장 좌석, 영화관 좌석
    - Icon을 하용한 Waffle 차트도 가능(Pictogram Chart)
- Venn [6:50 ~]
    - 집합(set)에서 사용하는 벤 다이어그램
    - 너무 많은 원을 그리면 가독성도 떨어지고 지저분해짐
- 실습 [9:05 ~]
    - MissingNo
    1. 타이타닉 데이터 셋 준비
    2. missingno [11:50 ~]
        결측치를 matrix로 나타내어 흰 부분으로 표시
        row당 결측치 개수가 다르기 때문에 sort로 정렬하면 보기 쉬움
        막대그래프로도 사용 가능
    - Treemap [13:45 ~]
        label - 텍스트 라벨(Pie 차트와 유사)
        color - 색을 개별적으로 지정 가능
        pad - 사각형에 마진을 주어 떨어뜨릴 수 있음
        text_kwargs - 텍스트 요소를 딕셔너리로 전달
    - Waffle Chart [16:55 ~]
    1. 기본 와플
        FigureClass = waffle
        rows - 행
        columns - 열
        valuew - 값(개수, 비율)
    2. legend
        딕셔너리로 전달
    3. color
        cmap_name - 컬러맵을 전달해서 색 변경
        colors - 개별적으로 범주의 색 전달 가능
    4. Block Arranging Style
        starting_location - 네 꼭지점을 기준으로 시작점 설정
        vertical - Defalut는 가로, 세로로 하려면 True
        block_arranging_style - 블록 나열 방식(기본은 snake방식)
    5. Icon
        icons - 아이콘 명칭
        icon_legend - 아이콘을 범례로 사용할 것인지
        font_size - 아이콘 사이즈
    - Venn [23:45 ~]
    1. 2개의 subset (이진법으로 지정)
    2. 3개의 subset (이진법으로 지정)
    3. set으로 전달 (교집합 count 자동 계산)

# (6-1강) Interactive Visualilzation

## 1. Interactive를 사용하는 이유 [00:26]

- 1.1. 정적 시각화의 단점
    - 공간적 낭비
    - 사용자마다 원하는 인사이트 상이
- 1.2. 인터랙티브의 종류
- 1.3. 라이브러리 소개

## 2. Interactive Viz Library

- 2.1. Matplotlib
- 2.2. Plotly
- 2.3. Plotly Express
- 2.4. Bokeh
- 3.5. Altair

# (6-2강) Interactive Visualization 실습

## 1. Plotly Express : Install Plotly [01:50]

## 2. Scatter, Bar, Line [03:00]

### 2-1. Scatter [03:00]

- x, y
- size, color
- range_x, range_y
- marginal_x, marginal_y
- hover_data, hover_name
- trendline
- facet_col, facet_row

### 2-2. Line [10:03]

- (seaborn과 비슷)

### 2-3. Bar [10:40]

- barmode

## 3. 다양한 차트 [12:44]

### 3-1. Part-of-Whole [13:33]

- Sunburst
- Treemap

### 3-2. 3-Dimensional [16:03]

- scatter_3d
    - symbol

### 3-3. Multidimensional [17:24]

- parallel_coordinates
- parallel_categories

### 3-4. Geo [19:30]

- scatter_geo
    - animation_frame, projection
- choropleth
    - projection

## 4. Plotly Interaction [22:40]

- button
- button list
- animation
- zoom in

# (7-1강) Custom Matplotlib Theme

## 1. 색의 선정 [01:10]

- color palette
- sns.palplot
- mpl.rcParams

## 2. Facet + Dark Mode 예시 [07:10]

- 2-1. Scatter plot
- 2-2. KDE plot
- 2-3. Pairplot
- 2-4. Plotly 3D plot

# (7-2강) Image & Text Visualization Tecniques

## 1. 비정형 데이터셋 EDA & Visualization [00:10]

- dataset meta data visualization
- dataset listup
- visual analytics
- train/inference visualizaion
- etc
    - XAI
    - node-link diagram
- 고민할 것
    - Interaction의 필요성
    - 데이터셋의 배치
    - 적절한 색, 투명도 사용

## 2. Image Dataset Visualization [08:40]

- 2-1. 이미지 나열 (ax.imshow)
- 2-2. Patch 사용하기 (mpl.patches)
- 예시. segmentation
    - 겹쳐서 투명도 조정
    - plotly botton 사용
    - wandb 사용
- 2-3. Dimension Reduction + Scatter Plot

## 3. Text Dataset Visualization [18:15]

- 3-1. Console output에 Highlight
- 3-2. IPython의 HTML 활용

## 4. Further Reading [24:15]


# ****(1강) Intro to NLP****

- 강의: 다양한 자연어 처리 task 및 최근 동향 소개, 자연어 처리 기본 알고리즘인 Bag-of-Words 소개 및 응용 예시 소개
- 실습: Naive Bayes classifier 구현, NLP를 위한 말뭉치 정제 작업 (Corpus cleaning) 실습
- 

## 1.  ****Intro to NLP****

***0:00*** ~ 머신러닝 대략적인 분야

1. NLP- Natural Language processing
    - tokenization
    - stemming어간추출
2. Text mining 
빅데이터 분야의 트렌드 분석 용도 등
3. information retrival 
추천 시스템
4. Treands of NLP 

***20:28*** ~ Bag-of-Words

1. Bag-of-Words Representation
    1. Bag-of-Words 의 구현 step 
2. NaiveBayes Classifier for Document Classification
    1. Bag-of-Words 의 벡터화된 문서를 분류하는 대표적 방법

# ****(2강) Word Embedding****

- 강의: one-hot-encoding 이외에 단어를 embedding할 수 있는 방법 중 Word2Vec과 GloVe에 대해 소개
- 실습: Word2Vec 구현 및 Embedding 시각화, 다국어 임베딩 실습
- 기본 과제1: Data Preprocessing & Tokenization(Spacy와 Okt를 활용한 영어/한국어 전처리 및 토큰화 기법)

- Word Embedding
- Word2Vec
- Distributed Representation

## 2. Word Embedding

1. 0:00~ What is embedding? 임베딩이란? 
    1. 단어들 사이의 관계를 잘 표현하기 위한 벡터화
    
2. 3:39 ~ Word2Vec
    1. 인접한 문장 끼리 관계성이 있을것이라 추정하는것
    2. Idea of Word2Vec
    3. How Word2Vec Algorithm Works 
    
    (무수한 설명과 필기와 공식)
    
3. 27:51 ~ GloVe : Another Word Embbedding Model
    1. 각 입력-출력 단어쌍의 관계에 대해 사전계산을 함
    2. 내적값을 구하는 새로운 loss function 사용
    
    (그리고 무수한 설명과 자료와 공식)
    

# 기본과제 1 Data Preprocessing_Tokenization

## **1. 파이썬 기본 코드를 이용한 영어 텍스트 토큰화 및 전처리**

- **1-A) 토큰화기 (tokenizer) 구현**
- **1-B) Vocabulary 만들기**
- **1-C) 인코딩 및 디코딩**

## **2. [Spacy](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fspacy.io%2F)를 이용한 영어 텍스트 토큰화 및 전처리**

- **2-A) Spacy 활용법**
- **2-B) Spacy를 활용한 전처리 및 토큰화**

## **3. [Konlpy](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkonlpy.org%2Fko%2Flatest%2F)를 활용한 한국어 토큰화**

# ****(3강) Basics of Recurrent Neural Network****


### Basics of Recurrent Neural Networks[00:25 ~]

- 서로 다른 time step에서 들어오는 입력 데이터를 처리할 때, 매 time step 마다 동일한 파라미터를 가지는 RNN 모델을 사용
- 이전 시점의 hidden state h_t-1과 t에서의 입력벡터 x_t가 input으로 들어가고, 현재 시점에서의 hidden state vector가 출력됨

$$
h_t = f_w(h_{t-1} ,x_t)
$$

- y_t : t에서의 output vector → task에 따라 최종 output을 계산할 때만 계산할수도, 매 time step마다 계산할 수도!
- hidden state 계산 수식 및 그림
    - 차원 이해하기!
    - 가중치 행렬을 통해 x_t와 h_t-1을 h_t로 변환해줌

### Types of RNNs [13: 08~]

- **one to one [ 13:20 ~ ]** : 입력이 시퀀스 데이터가 아닌 일반적인 구조 ( standard neural network)
- **one-to-many [ 14:42 ~ ]** : 입력은 하나의 time step이고, 출력은 여러개의 time step ( ex. image captioning )
- **many-to-one [ 16:15 ~ ]** : 입력은 sequence 데이터이고, 최종값은 마지막 time step에서만! (ex. sentiment classification)
- **many-to-many [17:51 ~ ]** : 입력과 출력이 모두 sequence 형태
    - (ex1. machine translation) : 입력 문장을 다 읽고 나서 출력 (delay 존재)
    - (ex2. video classification on frame level) : 입력이 주어질 때마다 입력 수행 (delay 존재x)

### Character-level Language Model [20:24~]

: 문자열의 순서를 바탕으로 그 다음 어떤 글자가 올지 맞추는 task

1. character level 사전 구축 
2. 각각의 character는 총 사전의 개수만큼의 dimension을 가지는 원핫벡터로 표현 가능
- training 과정
    
    $$
    h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t + b) 
    $$
    
    - W_hh : h_t-1에서 h_t로의 선형변환
    - W_xh : h_t 에서 h_t로의 선형변환
    
    → 그 다음에 나올 character를 예측하고, ground truth와 비교하며 학습 진행
    
    $$
    Logit = W_{hy}h_t + b 
    $$
    
    - logit : 사전에서 정의된 4개의 character 중 그 다음에 나올 character의 확률
        
        → ground truth에 해당하는 character의 확률을 높여야 함!
        
- inference 과정[30:29~]
    
    : 첫번째 character를 입력으로 준 후, 예측값으로 다음 time step의 character를 얻어내고, 다시 그것을 다음 time step의 입력으로 재사용
    
- 학습 예시

### Backpropagation through time (BPTT) [39:29 ~]

: character level Language Model의 학습 과정

- 한정된 computing power로 인해, 전체 sequence를 학습 시키는 것이 아니라, 제한된 길이로 sequence를 학습
- back propagation을 통해 W_hh, W_hy, W_xh가 학습됨
- how rnn works : hidden state의 특정 dimension을 고정해놓고 해당 dimension 값이 어떻게 바뀌는지 시각화하면, 그 역할을 파악할 수 있음

### Vanishing/ Exploding Gradient Problem in RNN [46:39 ~]

- 동일한 행렬(W_hh)가 반복적으로 곱해지면서 gradient가 기하급수적으로 증가 / 감소

# (4강) LSTM, GRU

## Long Short-Term Memory (LSTM)

### LSTM의 개념 [00:55]

- 간략한 그림 및 수식
- Cell state vector: 보다 완전한 정보 담고 있음
- Hidden state vector: 노출할만한 정보만 담긴 필터링된 벡터

### LSTM gate [04:55]

- Input Gate
- Forget Gate
- Output Gate
- Gate gate
- $x_t$와 $h_{t-1}$이 어떻게 concate되는가?
- 각 게이트 별 연산 방법 및 활성화함수 소개

### Detailed Process [08:40]

- 각 게이트 별 디테일한 그림 및 연산 설명
- Cell state와 Hidden State 연산이 어떻게 이뤄지는지 설명

## Gated Recurrent Unit (GRU)

### GRU의 개념 [17:50]

- GRU는 LSTM과 크게 유사하지만, 경량화에서 압도적 장점!
- hidden state(LSTM의 cell state의 역할)만을 가짐
- 두 개의 독립된 gate에서 하던 연산을 하나에서만 수행

### Backpropagation in LSTM, GRU [23:00]

- 필요한 정보를 곱셈이 아니라 덧셈의 형태로 하면서, Gradient Vanishing 문제 사라짐
- Long-term dependency 문제 해결

# 기본과제 2 RNN-based Language Model

## 데이터 업로드

## 데이터 클래스 준비

- Dictionary, Corpus의 개념

## 모델 아키텍처 준비

- RNNModel

## 모델 학습

- 배치화
- **Truncated Backpropagation through Time (Truncated BPTT)**
- 모델 학습
- 모델 평가

# Day 20 회고

- 무려 4일을 학습하지 못했다. 어떻게 강의 내용은 따라갔는데, 코드를 자세히 보지 못했다. AI 엔지니어가 코드를 못 다룬다는 것이 어불성설이기 때문에, 이번주 주말은 반드시 코드 리뷰를 통해 평일해 따라가지 못했던 내용을 따라가야겠다.
- 초반 1,2주 동안 팀의 학습 시스템을 만드는데 많은 시행착오를 거쳤다. 당장은 불필요해 보이는 작업들이 후에 내가 낙오자가 되는 것을 막아줬다. 열심히 챙겨준 다른 팀원들에게 감사함을 느낀 한 주!