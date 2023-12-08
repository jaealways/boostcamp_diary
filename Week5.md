# 5. Seq2Seq

## Seq2Seq with attention

### Seq2Seq Model [00:20]

- RNN 구조 중 many to many에 해당
- 입력 sequence를 모두 읽은 후 출력 sequence를 예측
- 인코더: 입력 문장을 읽는 RNN
- 디코더: 출력 문장을 생성하는 RNN
- 인코더와 디코더는 별도의 파라미터를 사용
- 디코더에서 <SoS>부터 생성 시작, <EoS>에 생성 종료
- hidden state vector의 사이즈가 고정되어 있기 때문에, 긴 인풋 들어와도 이 안에 모든 정보를 압축해서 담아야 함
- LSTM을 지나가면서 맨 처음 정보가 소실되거나 변질됨
- 입력문장의 순서를 뒤집는 트릭 사용! I go home → Home go I, “나는”부터 제대로 번역되게…

## Encoder-decoder architecture

### Attention [08:15]

- 어텐션 사용하면 마지막 hidden vector 뿐만 아니라, 전체를 제공해서 모든 단어의 특성 잘 반영하게 함
    - Are you ~? 에서 h2는 you에 관한 정보 잘 포함함, 모든 단어에 해당하는 h를 제공
- Decoder Hidden Vector + Encoder Hidden Vector 세트가 Attention 모듈의 인풋으로 들어감
- Encoder Hidden State Vector의 가중평균된 값이 아웃풋으로 나오게 됨
- 첫 단어 예측 이후 다음 단어 예측할 때, 기본적으로 같은 모듈을 사용하되, h2 hidden state vector를 통해 내적 → 앞의 경우와 다른 값 나오고, 다른 아웃풋 나옴
- 합이 1인 형태의 가중치 벡터를 attention vector라고 부름
- 이런 과정을 3,4번 째 단어를 예측할 때 EOS 나올 때까지 계속 사용함

### Hidden State Vector [19:40]

- Decoder 속 hidden state vector의 역할
    - 해당 state의 output 단어의 입력으로 사용
    - Encoder 속 어떤 단어 벡터를 중점적으로 가져와야 할지 Attention 가중치를 결정해주는 역할
- 단어가 잘못 선택된 경우, hidden state vector를 통한 backprop 진행

### Teacher Forcing [21:50]

- Teacher Forcing: 모델을 학습시킬 때 잘못된 단어를 예측하면, 원래 ground truth 값을 넣어줘서 다음 step 학습에 반영하는 것
    - 정확도가 개선되는 장점이 있지만, 실제 환경과는 괴리되는 모습 (실제 환경에선 실제 아웃풋을 다음 step에 넣어주기 때문…)
    - 처음에는 teacher forcing 사용하다가, 모델이 어느 정도 정확도 보이면 점점 줄여나가서 실제 환경과 유사하게 학습시키는 방법 있음

## Attention mechanism

### 다른 유사도 계산법 [26:00]

- hidden state vector의 유사도를 구하는 방식을 내적을 통해 수행했는데, 다른 방법 사용할 수 있음
- 인코더에서 각 단어의 h 벡터와 디코더 step에서의 h 벡터를 계산하는 방법
    - $h_t^T \bar{h_s}$: dot product, 내적 연산 수행 dot
    - $h_t^T W_a \bar{h_s}$:  가운데 가중치 행렬을 추가해서 내적 연산에 변형을 줌 general
    - $v_a^T \textbf{tanh}(W_a[h_t;\bar{h_s}])$: concat, 추가적인 학습이 필요한 파라미터가 end to end learning을 통해 학습(?)

### Attention의 장점 [38:30]

- Attention이 s2s 모델에 추가되면서 기계번역 성능 상당히 개선
- 학습의 관점에서 gradient vanishing 문제를 해결
    - Attention이 없었다면 이전 hidden state를 모두 거치면서 값이 희석되었을 것
    - 정보에 바로 접근할 수 있는 path 만들어짐
- 어텐션은 해석가능성을 높임. 어텐션의 분포를 조사해서, 디코더에서 어떤 인코더의 단어에 집중했는지 확인 가능
- 언제 어떤 단어 봐야하는지 (alignment)를 뉴럴넷이 스스로 배움

### Attention 기계번역 예제 [42:45]

- Attention을 통해 기계번역시 output 단어가 어떤 단어 참조했는지 확인 가능
- la zone economique europeenne → area economic european 순으로 참조해서 원래 인풋과 역전된 방향으로 참조하는 것 확인
- 한 단어 → 여러 단어 디코딩, 여러 단어 → 한 단어 디코딩 등 모두 가능

# 6. Beam Search and BLEU score

## Beam Search [00:20]

### 1. Greedy Decoding
- 현재 timestep에서만 가장 좋아보이는 단어 선택 (전체적인 맥락 못 봄)
- **core idea** : timestep t 일 때, t-1 단어 기준 높은 확률의 단어 선택
- **cons** : no way to undo

### 2. Exhaustive Search

- **core idea** : timestep t 일 때, 0~t-1 단어 기준 높은 확률의 단어 선택
- **cons** : expensive complexity (VocabSize^TimeStep)

### 3. Beam Search

- **core idea**
    
    timstep t 일 때, 0~t-1 단어 기준 높은 확률의 단어 선택하되 
    
    → 매 timestep 마다 k개(=beam size) 의 hypotheses(가설, 경로) 만 트래킹
    - Greedy Decoding과 Exhaustive Search의 중간
    
- **cons** : global optimal 보장 X
- **pros** : Exhaustive Search 보다 효율적
- **Stopping Criterion**
    - 1개의 hypotheses 종료 시점
        - <EOS> 가 나왔을 때
    - 전체 종료 시점
        - timestep T 일 때
        - n 개의 hypotheses 완료됐을 때
- **Finishing Up**
    - Normalize by Length (1/t)
        - 긴 timestep을 가지는 hypothesis 일수록 낮은 score를 가지기 때문 (log P < 0)
    - 가장 높은 score의 hypothesis 선택

## BLEU score [19:30]

### 1. F_measure

- I love you, Oh I love you는 유사도 높은데, 앞에서부터 one by one으로 비교하면 하나도 매칭 안됨
- 이를 해결하기 위해 measure에 대해 고민해야 함

- Precision
    - #correction words/len(prediction)
    - 검색엔진에서 자주 사용
- Recall
    - #correction words/len(reference)
- F_measure
    - Precision, Recall의 조화평균
    - 산술평균 ≥ 기하평균 ≥ 조화평균
- cons : 순서 고려 X

### 2. BLEU score

- 단순히 몇 개의 단어가 맞는지만을 갖고는 문법적으로 말 되는지 고려 못해줌
- 이를 위한 대안으로 BLEU score 제안
- N개의 문구가 얼마나 ground truth와 겹치는가?
- recall 대신 precision만 고려
    - I love this movie very much -> 나는 이 노래를 많이 사랑한다 (한 단어만 틀려도 명백한 오역)

- N-gram : N개(=1~4개) 의 단어 배열 고려
- precision : precision들의 기하평균 계산, 낮은 값을 훨씬 더 크게 반영하겠다
- brevity penalty : recall 역할 추가, 문장이 ground truth 보다 짧으면 그만큼 precision 값 낮추어주겠다

# 7. Transformer 1

### Transformer High-Level-View [00:30]

- Attention is all you need
    - Attention만을 사용해서 RNN을 통째로 대체할 수 있다. [01:05]
- RNN: Long-Term Dependency [01:25]
    - 초기의 단어를 전달하려면 많은 time step을 거쳐야 한다. [04:00]
    - Bi-Directional RNNs [04:40]
    - Forward와 Backward의 hidden state 값을 concat함!
- Transformer Long-Term Dependency [08:00]
    - input 시퀀스 전체 내용을 반영한 벡터(x)가 → output으로 나오게 됨(h) [09:00]
    - 어떻게 어텐션으로 각 단어 시퀀스를 보고 인코딩 벡터를 만들어내는지 [09:30]
    - Query, Key, Value 벡터의 의미 [15:00]
    - Wq, Wk, Wb를 통해 변환 된 벡터를 생각하고 attention 모듈 적용 [18:30]
        - q는 하나로 고정되어 있어도, k와 v는 서로 개수가 동일해야 함 [19:55]
    - 마찬가지로 두 번째 단어(go) attention 모듈 적용 [22:20]
        - key와 value는 동일하되, query를 go단어와 Wq로 변환을 한 q벡터를 적용 [22:30]
    - 오른쪽 그림 : 다른 예시를 행렬 연산 관점에서 보기 [23:00]
    - 각 단어의 인코딩 벡터를 구할 때, 정보가 어떤 방식으로 반영되는지 [26:00]
    - 시퀀스가 길어서 time step 갭이 커도 동일한 k, v로 변환이 되어서 반영 가능 [27:05]
    - 결론 : Self-Attention모듈은 Long-Term Dependency문제를 근본적으로 해결! [28:20]

### Scaled Dot-Product Attention [28:35]

- 어텐션 모듈의 과정을 수식적으로 살펴봄 [28:45]
    - Q와 K 벡터 차원은 동일, V는 차원이 달라도 됨 + 이유 [29:45]
    - 어텐션 모듈 계산 과정 [30:55]
- 어텐션 모듈의 입력과 출력 관계 (행렬 크기에 집중) [33:15]
    - 여러 쿼리 벡터에 대한 어텐션 모델 연산을 행렬 연산으로 → 병렬화 계산에 특화된 gpu를 활용 → RNN에 비 효율적이고 빠르게 수행 가능 [43:35]
- 두 단어(Thinking Machines)로 이루어진 시퀀스 인코딩 과정 그림으로 살펴보기 [44:10]
- Q, K 내적 계산 과정에서, scaling에 대해(루트 dk로 나누는 연산) [48:00]
    - 차원 수가 커지면,  각 벡터의 원소에 대한 분산이 커짐 [48:40]
    - 분산(표준편차) 클수록, softmax 확률 분포가 큰 값에 몰리는 것을 알 수 있음 [54:30]
    - 분산이 작은 경우, 확률분포가 고르게 나오는 것을 알 수 있음
    - 내적에 참여하는 Q, K의 차원의 수에 따라  내적 값의 분산이 좌지우지 될 수 있고, 이에 따라 softmax 확률 분포가 특정 원소에 몰리는 패턴이 나올 수 있음  [55:05]
    - 루트 dk로 나누는 이유 : 분산을 일정하게 유지시킴으로써 학습을 안정화 [55:45]
    - 결론 : softmax가 한곳으로 몰리는 경우, gradient vanishing이 발생할 위험이 높으므로, softmax output을 적절한 범위로 조정하는 것이 학습에 중요함 [57:30]


# 8. Transformer 2

### Multi-Head Attention [00:18 ~]

: 동일한 Q,K vector들에 대해 동시에 병렬적으로 여러 버전의 attention 수행

- **Multi-Head Attention이 필요한 이유 [02:51]** : 동일한 시퀀스가 주어졌을 때에도 어떤 특정한 쿼리 word에 대해서 **서로 다른 기준으로 여러 측면에서의 정보**를 뽑아와야할 필요가 있음
- Example from illustrated transformer [04:27]
    - multi head vector들을 단순히 concat
    - 차원이 늘어남 → 별도의 dimension을 줄여주는 역할을 하는 W 곱해줌!
- Multi-head Attention 연산량 [06:53]
    - GPU 코어수가 충분한 상황에선 병렬화가 가능하므로, RNN보다 훨씬 빠르게 계산 가능하지만, RNN보다 많은 양의 메모리 요구

### Components of Transformer [17:10 ~]

: 추가적인 후처리 진행해서 하나의 모듈로 작동

- **Residual Connection** 후 Layer Normalization 수행 : gradient vanishing 해결, 학습 안정화
    - LayerNorm(x+sublayer(x))
- **Layer Normalization [22:34~]**
    - **Step1.** 각 word별 평균 0, 분산 1로 맞춰주기
    - **Step2.** 각 노드별 Affine Transformation ****(y = ax+b) 적용
- **Positional Encoding [31:02 ~]**
    - Transformer에선, 단어의 순서관계 구별 x
    - 각 순서를 특정 지을 수 있는 상수 벡터를 워드 입력벡터에 더해줌
    - dimension에 따라 각각의 sin cos 함수를 번갈아 사용해주면서 생성
- **Warm-up Learning Rate Scheduler [40:31 ~]**
    - learning rate를 학습 중 적절히 변경! → 경험적으로 좀 더 좋은 성능
- High-Level View
- Encoder Self Attention Visualization
- **Decoder[50:53~]**
    - Masked Multi- Head Attention : ground truth에서 한칸씩 shift된 벡터들을 입력으로 받음!
    - Multi-head Attention : 인코더의 최종 출력을 Key와 Value로, 디코더에서 만든 hidden state vector를 Query로 사용 → 인코더 디코더간의 attention module
    - linear : target language vocabulary size에 맞게 벡터 변환
- **Masked Multi-Head Attention [ 56:32~]**
    - 예측과정에선, 다음 sequence의 등장하는 단어들은 알 수 x → 다음 단어들은 0으로 mask해준 후 attention module 적용!
- **Experimental Results [1:01:12]**

# 기본 과제 4 Preprocessing for NMT Model

# 9. Self-supervised Pre-training Models

- GPT-1
- BERT

### Self-supervised Pre-training Models , Recent Trends 최근동향 [00:46~]

- 범용적인 인코더, 디코더 로서의 Transfomer 가 활발하게 사용중.
- self-attention 블록을 더 많이 쌓은 형태도 나타남
- Greedy decoding

### GPT-1 [02:52~]

introduces special tokens

- 순차적으로 다음 단어를 예측하는 language model
- 다수의 문장이 존재하는 경우에도 활용 가능한 frame work 제안. 부정과 긍정을 분류하는 감정분류를 학습 가능.
- classification , Entailment , Similary , Multiple Choice 의 task 를 self-attention 블록을 이용하여 수행
    - pre-train 된 지식들을 잃지 않기위해 task 별로 마지막 layer 를 변경하여 사용
    - 대규모 데이터로 부터 얻은 지식을 기반으로 소규모 데이터에 적용하기 위해 사용

### BERT [13:45~]

다음 단어를 예측하기 위한 pre-training 

- ELMo : 트랜스포머 이전에 LSTM 을 이용한 다음 단어 예측 모델
- GPT-1 의 경우 <SOS> 토큰을 이용하여 전통적인 방식으로, 전후 문맥을 보지 않고 전 문맥을 기반으로 예측한다는 문제점이 있었음
    - BERT 는 전후(좌우) 문맥을 고려하여 단어를 예측 할 수 있다.

BERT 의 학습 방식들 [16:47~]

- MLM (Masked Language Model)
    - bert 는 15% 의 비율을 mask 로 치환하여 단어 예측을 학습 한다.
    - 단점도 존재. 15% 의 비율과 해당 단어들에 익숙한 모델이 나올 수 있음. 때문에 해당 15% 의 단어 내부에서 또 변화를 주기도 함
    - 또는 10% 만 사용
- Next Sentence Prediction
    - 문장들 간의 관계 예측을 위해, 주어진 문장 B 가 문장 A 를 잇는 실제 문장인지 아닌지 예측하는 방식

BERT 의 구조 [25:35~]

- self-attention 구조를 그대로 가지되 블록을 몇개를 쌓았느냐로 Base 모델(12) 과 LARGE 모델(24) 이 나뉨
- Transformer : positional Encoding
- segment Embedding

BERT 와 GPT-1 의 차이 [31:40~]

- GPT-1 은 Masked Multi Self-Attention 을 사용한다
- BERT 는 Mask 로 치환된 단어들을 예측하게 되고, 때문에 전후 모든 단어들에 접근이 가능하며 트랜스포머에서 인코더에 사용되던 Self-Attention 을 사용한다.

BERT 의 Pine-Tuning 방식 [33:31~]

- 사전학습한 모델을 새로운것에 적용시켜 사용한다
- 문장 연관성, 단일 문장 분류, 질의응답, 문장 성분 분석 등의 task 가 있음

BERT 와 GPT-1 의 차이 [35:40~]

- 데이터 사이즈부터 다르다! BERT 가 사용한 학습데이터가 더 많다!
- SOS 토큰이 아닌 [SEP],[CLS] 토큰사용. Segment embedding 사용!
- 

# 10. Advanced Self-supervised Pre-training Models

## GPT-2 [0:55 ~]

- 간략한 설명 [1:00 ~]
- Transformer 기반의 모델
- 40GB 텍스트 학습 → 데이터 품질이 높은 것 위주로 사용
- zero-shot setting
- Motivation [3:55 ~]
Multitask Learning as Question Answering 논문
- Datasets [6:40 ~]
고품질의 데이터를 선별적으로 배우도록 Reddit 플랫폼 스크랩
- 외부 링크를 포함하고 좋아요 3개 이상의 글 학습

- Preprocess [8:40 ~]
  BPE (Byte pair encoding) 사용
- Modification [9:00 ~]
- Question Answering [10:00 ~]
- Summarization [11:40 ~]
TL;DR - 이 단어가 보이면 앞의 데이터를 요약 테스크 실행
- Translation [12:55 ~]

## GPT-3 [13:25 ~]

- 간략한 설명 [13:35 ~]
- GPT-2보다 훨씬 많은 parameter 수 → transformer의 self-attention 블럭 많이 쌓음
- 3.2M의 배치 사이즈
- zero shot에서 놀라운 성능
- 모델사이즈를 키울수록 zero, one, few shot의 성능이 높아짐

## ALBERT(A Lite BERT) [18:05 ~]

- 간략한 설명
- 많은 메모리, 긴 학습 시간을 해결하고자 성능 하락 없이 모델 사이즈 간소화
- Factorized Embedding Parameterization [19:25 ~]
임베딩 예시 설명 [19:55 ~]
→ 입력 임베딩 차원이 끝까지 유지되기 때문에 적절한 사이즈 선택 필요
→ 4차원 BERT 벡터(4x4)를, 입력(4x2) x 가중치(2x4)으로 쪼갬으로 크기 유지.
(사이즈가 크면 차이가 큼 Ex_ 500x100 ⇒ 500x15 + 15x100)
→ MHA의 특징 가짐.
- Cross-layer Parameter Sharing [30:30 ~]
- Shared-FFN(feed-forward network)
- Shared-attention
- all-shared (Parameter 가장 적음, 하지만 성능 큰 차이 X)
- Sentence Order Prediction [32:50 ~]
- Next Sentence Prediction의 존재 필요성이 떨어짐
- Negative samples : 서로 다른 문서의 문장을 가지고 와서 사용 → Next Sentence가 아님을 판별하기 쉬움
- GLUE 벤치마크 Results [40:35 ~]
- 다른 BERT 모델보다 ALBERT모델의 성능이 뛰어남.

## ELECTRA [41:40 ~]

(Efficiently Learning an Encoder that Classifies Token Replacements Accurately 논문)
- 단어 복원 모델 (Generator) : mask 된 단어를 원래 단어로 복원
- 판별 (Discriminator) : generator 모델이 준 단어가 original인지 replaced인지 판별
→ generator - BERT / Discriminator - Transformer기반 = GAN
→ pre-train모델로 사용 가능한 모델 : Discriminator

- 성능 비교 [46:45 ~]
mask모델과 비교했을 때 학습 량이 커질수록 GLUE score 성능 뛰어남

## 경량화 모델 [47:55 ~]

- DistillBERT [49:55 ~]
haggingface에서 발표한 논문
teacher model / student model
- TinyBERT [53:25 ~]
teacher model / student model - student 모델이 teacher 모델을 닮도록

## Fusing Knowlege Graph into Language Model [56:35 ~]

- Transformer + Graph(외부 정보, BERT의 약점 커버)

- ERNIE
- KagNET