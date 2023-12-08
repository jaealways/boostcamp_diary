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

