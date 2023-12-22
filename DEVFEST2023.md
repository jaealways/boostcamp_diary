## 효과적인 단위테스트
- Unit Testing 단위테스트: 생산성과 품질을 위한 단위 테스트 원칙과 패턴

### 단위 테스트의 목표
- 테스트 코드 없으면 코드 복장섭 커지면, 시간 복잡도가 기하급수적으로 증가할 수 있음
- 테스트 코드의 품질도 중요함!!, 작업 효율을 유지하는 것이 중요

### 단위 테스트의 정의
- Unit test, Intergration tests, UI tests (점점 통합됨)
    - 작은 부분들에 집중
    - 테스트 툴을 사용해서 프로그래머들이 직접 작성 (격리된 방식으로 처리??, 병렬로 가능)
    - 다른 테스트 방식에 비해 빨라야!
### 단위 테스트의 두 가지 관점
- 고전파
    - 이전 방식을 고수
    - 기능 단위
    - 외부 기능이거나 덜 중요한 것
- 런던파
    - 새로운 모킹 툴을 적극적으로 활용(?)
    - 코드 단위, 나머지 의존관계가 있으면 MOCK으로

### 좋은 단위 테스트의 4대 요소
- 버그방지
- 리팩토링 내성
    - 리팩토링 해도 단위 테스트에 영향 가면 안됨
    - False positive 최소화해야 함
- 빠른 피드백
- 유지 보수성
- 위의 네 가지를 점수화해서 곱해서 측정



##  텍스트-to-이미지 생성모델 요즘 이야기
- 몇 년 전엔 GAN 이전 이후, 요즘은 Diffusion 이전 이후로 나눔
- AI의 주요 관심사: 거대모델, 멀티모달, 경량화, 고품질, 커스텀
- 2015년만 해도 생성형은 정형 데이터 수준, 지금은 구분 불가
- Discriminative vs Generative

### GAN

- GAN 기반으로 엄청난 논문 쏟아짐 (2년 정도)
    - 비즈니스화까진 힘든 퀄리티...
    - GAN은 거대화의 한계, 파라미터 늘리면 학습의 불안정성
- 디퓨전 vs GAN
    - 디퓨전은 한 번에 생성 불가능 (아웃풋을 다시 인풋에 넣는 과정 최소 몇 십번 필요)
    - 디퓨전이 잠재력 더 큼. resnet이 잘되는 이유와 비슷. Denoising test를 학습함

### Stable diffusion 전과 후
- 출시 후 오픈소스화

### 2023 Q4
- SDXL: 입력과 출력을 키우고 SD 두 개를 붙힘

### 멀티모달 DALLE 3
- 이미지 캡션 많으로도 좋은 결과 일으킴
- 경령화 Turbo: 디퓨전은 추론 반복으로 속도가 단점, GAN 기반의 방식 사용해서 좋은 이미지 한 번에 만들 수 있었음
- 커스텀: 적은 인풋 사진으로도 원하는 화풍(고흐 등)의 사진 만들 수 있음 Dreambooth, Textual inversion, 
- 커스텀2: Lora, controlnet Text to image generation


##  감정을 이해하는 AI

### AI를 활용한 감정 분석
- 비전으로 표정 학습
- 표정 외에 Bio 정보도 응용할 수 있지 않을까?

### Brain Computer Interface
- 뇌와 컴퓨터 간의 통신을 가능하게 하는 시스템, 외부 장치에 뇌의 활동을 전달
- EEG: 사람 머리에 장치를 통해 감정을 파악할 수 있음
- Emotion Fingerprint: 감정지문, 사람마다 갖는 감정의 흔적(?), 어떻게 감정지문을 매핑할까
- 지역과 문화에 따라 느끼는 감정이 차이가 있을까?

### DEAP Dataset
- BCI 분석을 위해 사용
- 논문마다 기준이 다르고, 공개된 코드도 많이 없음
- 감정을 정확히 인지하고, 지표화하기 쉽지 않기 떄문에 데이터 품질이 낮음


### EEG
- Out-of-data 예시: 숫자 9를 좌우반전하면 8에 가장 높은 확률 부여
- OOD Detection model: OOD 성능평가 지표를 확보하기 위해(?)
- MSP: Max softmax probability

### Domain Shift
- 좋은 데이터 얻었을 떄, Gender 등 여러 팩터를 기준으로 분포의 특성 살펴봄
- WGANDA: wasserstein distance를 활용해서 GAN의 문제 풀기
    - 학습데이터와 평가데이터의 도메인 차이를 극복하기 위해 사용
- Domain Adaptation Neural Network: 한 쪽의 도메인을 다른 쪽의 도메인으로 맞추고자 함
- CapsNet: 자전거를 타고 있는 사람만으로, 자전거 속도가 빠르다고 추측함
    CNN은 Entity를 표현하기엔 수가 부족

### 외향적 요소 분속
- YOLO v8를 이용한 MRI 영상 분석
- B


##  Introduction to vector database

- semantic한 db 검색 가능 -> vector db 사용
    - notion 등은 키워드 위주 검색, 뭔가 부족?
    - 나만의 뇌를 갖는다고 생각??
- 사내 문서들을 

### Examples of NN (뉴럴넷 기반 검색)
- Brute Force: 연산량 엄청 많음
- K-means: 
- 차원의 저주: 2천차원 이상이면 의미있는 결과 얻기 힘듦
    - 결론은 인덱싱을 적절히 해야한다!

### Indexing 전략
- Inverted File, Graph Based, Tree-based 등
- 모든 엔지니어링은 trade-off, 벡터 서치에선 속도와 퀄리티
- Recall과 Latency의 트레이드오프
- IVF-FLAT: 역방향 색인
    - 단어를 찾았을 떄, 단어가 refer하는 문서가 무엇인가?
    - 순방향은 문서 -> 단어
- IVF-PQ
- HNSW: Hierarchical Navigable Small Worlds
    - 얼마나 layer 층 타고 들어가고, 얼마나 가까워야 정답으로 찍을 것인가?

## Graph Neural Network
- 그래프를 인접행렬로 표현 -> 추천 시스템, 화학 분자구조 등의 application
- GNN -> graph data -> graph update layer -> Neural Network -> outputs


### 그래프 거리 계산 방식


