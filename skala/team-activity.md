
(1) 모델 크기 (5.1 모델설정)

n_layer, n_head, n_embd를 줄이거나 늘려보기 (GPTConfig 셀)
"작은 모델도 문장을 외울까?"

-> T2-1, T3-2, T2-2

(2) 학습 반복 수 (2.실행설정)

MAX_ITERS를 100 / 500 / 2000 등으로 바꿔보기
바꿔가며 loss 곡선과 생성 문장 비교


(3) 문맥 길이 (block_size) (5.1 모델설정)

32 / 128 / 256 등 으로 바꿔보기
짧은 문맥과 긴 문맥이 생성 품질에 주는 영향 확인

-> T1-1

(4) 데이터 양 (input_kr.txt 파일 조정)

corpus 전체 대신 절반만 잘라서 학습해보기
데이터가 적을 때 나타나는 특징(암기 vs 일반화) 비교
-> T1-2

(5) causal mask 유무 비교 (5.3 Transformer Block)

Block 클래스 안의 self.attn = CausalSelfAttention(config)를 SelfAttention(config)로 바꿔보기.
mask 영향 보기, 미래를 미리 보면 어떻게 되는가?

-> T3-1

(6) 생성 파라미터 (temperature, top_k) (10. 학습 후 생성 결과)

값을 낮추면 안정적이지만 뻔한 문장, 높이면 다양한 문장
트레이드오프를 직접 눈으로 확인

-> T6-1, T6-2

