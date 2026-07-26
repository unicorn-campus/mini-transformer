# -*- coding: utf-8 -*-
"""미니 트랜스포머 공통 모듈 — `hands-on/example-explain.ipynb` 를 그대로 옮긴 것.

노트북의 클래스명·함수명·변수명을 **하나도 바꾸지 않았습니다**. 노트북과 이 파일을
나란히 놓고 읽으면 셀 하나가 어디로 갔는지 바로 찾을 수 있습니다.

노트북 셀 ↔ 이 파일 대응
    0단계 준비              → setup_korean_font()
    1단계 데이터            → DATA
    2단계 토큰화/단어장     → PAD/SOS/EOS/UNK, SPECIALS, tokenize(), Vocab
    3단계 부품 ①~⑩         → PositionalEncoding … make_decoder_mask
    4단계 배치 만들기       → build_batches()
    5단계 답 생성           → answer()
    6단계 어텐션 히트맵     → show_attention()

이 모듈은 부품만 제공합니다. 실제 학습은 `training.py`, 추론은 `reasoning.py` 에서 합니다.
"""
import logging
import math
import sys

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Windows 기본 콘솔(cp949)은 '✅' 같은 기호를 인코딩하지 못해 UnicodeEncodeError 로 죽습니다.
# 노트북(Colab/Jupyter)은 UTF-8 이라 문제가 없지만, 스크립트로 돌릴 때를 대비해 맞춰 둡니다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 0단계: 그래프 한글 폰트 준비 (노트북 셀 2)
# ─────────────────────────────────────────────────────────────────────────────
def setup_korean_font():
    """히트맵 축에 한글이 깨지지 않도록 폰트를 잡아 줍니다.

    Colab 이면 나눔고딕(`!apt-get install fonts-nanum` 으로 설치된 것)을 직접 등록하고,
    로컬이면 OS 에 흔한 한글 폰트를 순서대로 탐색합니다.
    """
    logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)  # 사소한 폰트 경고 숨김
    plt.rcParams["axes.unicode_minus"] = False                       # 마이너스 기호 깨짐 방지
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        plt.rcParams["font.family"] = "NanumGothic"                  # Colab: 나눔고딕
    except Exception:
        for _n in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:   # 로컬 폴백
            if _n in {f.name for f in font_manager.fontManager.ttflist}:
                plt.rcParams["font.family"] = _n
                break


# ─────────────────────────────────────────────────────────────────────────────
# 1단계: 아주 작은 '질문→답' 데이터 (노트북 셀 4)
# ─────────────────────────────────────────────────────────────────────────────
# 문장 '틀'은 똑같이 두고 '대상' 단어(먹구름/별/해/…)만 바꿨습니다.
# 답을 가르는 단서가 오직 그 한 단어뿐이라, 모델이 반드시 거기에 '주목'하게 됩니다.
DATA = [
    ("하늘에 먹구름이 보이면 뭐가 생각나", "비가 올 것 같아"),   # 핵심 쌍: '먹구름' → '비'
    ("하늘에 별이 보이면 뭐가 생각나", "밤이 깊었나 봐"),       # '별' → '밤'
    ("하늘에 해가 보이면 뭐가 생각나", "아침이 밝았구나"),       # '해' → '아침'
    ("하늘에 무지개가 보이면 뭐가 생각나", "비가 그쳤나 봐"),     # '무지개' → '비 그침'
    ("하늘에 눈송이가 보이면 뭐가 생각나", "겨울이 왔구나"),      # '눈송이' → '겨울'
    ("하늘에 노을이 보이면 뭐가 생각나", "저녁이 되었네"),        # '노을' → '저녁'
]


# ─────────────────────────────────────────────────────────────────────────────
# 2단계: 토큰화 & 단어장(Vocab) (노트북 셀 8)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/vocab.html
# ─────────────────────────────────────────────────────────────────────────────
# 특수 토큰 4종을 리스트 맨 앞 '예약석'에 두어 항상 같은 번호(0,1,2,3)를 갖게 합니다.
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"   # 특수 토큰
SPECIALS = [PAD, SOS, EOS, UNK]


def tokenize(s):
    """토큰화 = 문장을 공백 단위로 쪼개기. '먹구름이'는 통째로 한 단어입니다."""
    return s.strip().split()                    # 공백 단위로 자르기


class Vocab:
    """단어↔번호 양방향 사전. itos = 번호→단어(리스트), stoi = 단어→번호(딕셔너리)."""

    def __init__(self, sentences):
        # ① 모으기 → ② set 으로 중복 제거 → ③ sorted 로 정렬(재현성: '먹구름이'는 언제나 6번)
        toks = sorted({t for s in sentences for t in tokenize(s)})
        self.itos = SPECIALS + toks             # 번호 → 단어 (실제 단어는 4번부터)
        self.stoi = {t: i for i, t in enumerate(self.itos)}   # 단어 → 번호

    def __len__(self):
        return len(self.itos)                   # 임베딩 행 개수. 우리 예제는 14 / 18

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def sos_id(self):
        return self.stoi[SOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    def encode(self, s, add_special=True):
        """문장 → 번호 리스트. 모르는 단어는 <unk>(3번)로 대체."""
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(s)]
        # add_special=True 면 앞뒤에 <sos>…<eos> 를 붙여 '문장 시작/끝'을 알려 줍니다.
        return [self.sos_id] + ids + [self.eos_id] if add_special else ids

    def decode(self, ids):
        """번호 리스트 → 사람이 읽는 문장. 특수토큰은 신호일 뿐이라 걸러 냅니다."""
        return " ".join(self.itos[i] for i in ids if self.itos[i] not in (PAD, SOS, EOS))


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ①: 위치 인코딩 (노트북 셀 13)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/positional-encoding.html
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """단어 의미(임베딩)에 '몇 번째 자리인가'를 sin·cos 파도무늬로 더해 줍니다.

    학습 파라미터 0개 — 공식으로 값이 이미 정해져 있어 배울 것이 없습니다.
    """

    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        # 고정 무늬 하나에만 매달리지 않게 일부를 지워 줍니다(원논문도 같은 자리에 둡니다).
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                      # 빈 표 (22, 64)
        pos = torch.arange(0, max_len).float().unsqueeze(1)     # 위치 0,1,2,... 를 세로로
        # 차원마다 다른 파도 속도(주파수) 32개. 앞칸은 빠른 파도(옆 단어 구분), 뒤칸은 느린 파도.
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)     # 짝수 차원 = sin
        pe[:, 1::2] = torch.cos(pos * div)     # 홀수 차원 = cos
        # register_buffer: 학습은 안 되면서도 state_dict 저장과 .to(device) 이동에는 따라옵니다.
        self.register_buffer("pe", pe.unsqueeze(0))             # (1, max_len, d_model)

    def forward(self, x):
        # x = (B, S, d_model). 지금 문장 길이만큼 표를 잘라 브로드캐스팅으로 더합니다.
        return self.dropout(x + self.pe[:, :x.size(1)])   # 단어 의미 + 위치 파도


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ②: 어텐션 핵심 공식 (노트북 셀 15)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/attention.html
# ─────────────────────────────────────────────────────────────────────────────
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Q(검색어)와 K(색인)를 비교해 '누구를 몇 % 볼지' 정하고, 그 비율대로 V(내용)를 섞습니다.

    이 함수 안에는 학습되는 값이 하나도 없습니다(그래서 nn.Module 이 아니라 그냥 def).
    q, k, v = (B, h, S, d_k). q 와 k 의 S 는 달라도 되지만, k 와 v 의 S 는 같아야 합니다.
    """
    d_k = q.size(-1)                                                 # 벡터 한 개의 칸 수(16)
    # √d_k 로 나누는 이유: 칸이 많으면 점수가 저절로 커져 softmax 가 포화되고 기울기가 사라집니다.
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # 관련도 점수
    if mask is not None:
        # 0 으로 바꾸면 exp(0)=1 이라 오히려 큰 비율을 받습니다. 반드시 −무한대로 덮어씁니다.
        scores = scores.masked_fill(mask == 0, float("-inf"))        # 가릴 곳은 -무한대
    # dim=-1 = Key 방향. "답 단어 하나가 질문 단어들에 나눠 준 비율의 합 = 1"
    attn = torch.softmax(scores, dim=-1)                             # 합=1 비율로
    if dropout is not None:
        attn = dropout(attn)                                         # 학습 중에만 일부 연결을 끔
    return torch.matmul(attn, v), attn                               # V를 비율대로 가중합


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ③: 멀티헤드 어텐션 (노트북 셀 18)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/multi-head-attention.html
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """카메라 4대로 같은 장면을 다른 각도에서 찍고, 마지막에 한 장으로 합칩니다.

    들어갈 때 (B, S, 64) → 나올 때 (B, S, 64). 모양이 그대로라 몇 겹이든 쌓을 수 있습니다.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # h = 헤드(카메라) 수, dk = 헤드 하나가 맡는 칸 수. d_model 은 num_heads 로 나눠떨어져야 합니다.
        self.h, self.dk = num_heads, d_model // num_heads
        # 학습되는 건 이 변환기 네 대뿐입니다. wq 는 딱 한 대이고, 그 결과 64칸을 16칸씩 잘라 쓰는 것이 헤드입니다.
        self.wq = nn.Linear(d_model, d_model); self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model); self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None                 # 시각화용 보관

    def split(self, x):
        """64칸을 헤드별로 자르기. (B, S, 64) → (B, 헤드, S, dk)

        ⚠️ view(b, self.h, s, self.dk) 로 한 번에 쓰면 에러 없이 값이 엉뚱하게 섞입니다.
        """
        b, s, _ = x.shape
        return x.view(b, s, self.h, self.dk).transpose(1, 2)   # (B, 헤드, 길이, dk)

    def merge(self, x):
        """split 을 거꾸로. 헤드 조각을 다시 64칸으로 이어붙입니다(섞는 일은 wo 가 합니다)."""
        b, _, s, _ = x.shape
        # transpose 는 '보는 순서'만 바꾸므로, view 전에 contiguous() 로 한 번 정리해 줍니다.
        return x.transpose(1, 2).contiguous().view(b, s, self.h * self.dk)

    def forward(self, q, k, v, mask=None):
        # 값을 바꾸는 건 wq·wk·wv 뿐입니다. 헤드마다 다르게 보이는 이유가 여기 있습니다.
        q, k, v = self.split(self.wq(q)), self.split(self.wk(k)), self.split(self.wv(v))
        # for 문이 없는데도 헤드 전부가 계산됩니다. 앞쪽 (B, h) 축이 '묶음'으로 함께 처리되기 때문입니다.
        ctx, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        # detach() 로 '계산 과정 기록'을 떼어 메모리를 붙잡지 않게 합니다.
        # ⚠️ 히트맵은 model.eval() 에서 그리세요. train() 이면 드롭아웃 때문에 행 합이 1이 아닙니다.
        self.last_attn_weights = attn.detach()
        # merge 로 이어붙인 뒤 wo 로 한 번 섞어 내보냅니다(안 섞으면 헤드 경계가 그대로 남습니다).
        return self.wo(self.merge(ctx))


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ④: FFN (노트북 셀 21)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/feed-forward.html
# ─────────────────────────────────────────────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    """회의(어텐션)가 끝나고, 각자 책상에서 혼자 곱씹는 단계입니다.

    확대(64→256) → ReLU → Dropout → 축소(256→64). position-wise = 모든 단어가 '똑같은'
    신경망을 쓰지만 서로의 결과는 절대 보지 않는다는 뜻입니다.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # ⭐ ReLU 가 없으면 Linear 두 개가 '하나의 Linear'와 완전히 같아집니다.
        #    굳이 넓혔다 좁히는 이유는, 그 사이에 ReLU 를 끼워 넣기 위해서입니다.
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        # (B, S, 64) → (B, S, 64). Linear 가 마지막 축만 보고 일하므로 단어끼리 섞이지 않습니다.
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑤: 인코더 블록 (노트북 셀 23)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/encoder-layer.html
# ─────────────────────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """밑줄 긋고 → 혼자 곱씹고 → 원본은 안 버리는 '복습 한 장'.

    하위층 2개(셀프어텐션·FFN), 각각 뒤에 (원본 더하기 + 정규화)가 붙습니다(Post-LN).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # '셀프'인 이유는 forward 에서 q·k·v 자리에 모두 같은 x 를 넣기 때문입니다.
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 하위층이 2개니까 정규화도 2개. 단어 하나의 64칸을 평균 0·표준편차 1로 정돈합니다.
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)   # 드롭아웃 1개를 두 하위층이 함께 씁니다

    def forward(self, x, mask):
        # mask 는 '패딩 마스크'뿐입니다. 질문 문장은 이미 전부 갖고 있으니 미래를 가릴 이유가 없습니다.
        # 덧셈(잔차)이라 역전파 때 기울기가 그대로 지나가는 지름길이 됩니다.
        x = self.n1(x + self.drop(self.attn(x, x, x, mask)))   # 잔차 + 정규화
        x = self.n2(x + self.drop(self.ffn(x)))
        return x                                               # 들어온 모양과 똑같은 (B, S, 64)


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑥: 디코더 블록 (노트북 셀 25)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/decoder-block.html
# ─────────────────────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """답안지 뒷장은 못 보고, 문제지는 계속 곁눈질하는 수험생.

    하위층 3개 — 마스크드 셀프어텐션 → 크로스어텐션(질문 곁눈질) → FFN.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)   # 지금까지 쓴 답 돌아보기
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Q=답, K/V=질문 인코딩
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 하위층이 3개라 LayerNorm 도 3개.
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model); self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, tgt_mask, src_mask):
        # tgt_mask 가 '미래 단어 + 패딩'을 가려 정답을 미리 훔쳐보지 못하게 합니다.
        x = self.n1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))       # 미래 가림
        # q 와 k·v 의 길이가 다른 유일한 곳입니다 — q 는 답, k·v 는 질문.
        x = self.n2(x + self.drop(self.cross_attn(x, enc, enc, src_mask)))  # 질문 곁눈질
        x = self.n3(x + self.drop(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑦: 인코더 (노트북 셀 27)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/encoder.html
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """원고를 여러 번 다시 읽으며 질문에 대한 이해를 쌓는 단계.

    임베딩 × √d_model → 위치인코딩 → EncoderLayer × num_layers.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model                      # forward 의 √d_model 스케일링에서 씁니다
        self.emb = nn.Embedding(vocab_size, d_model)     # 단어 번호 → d_model 차원 벡터
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        # ⚠️ 파이썬 리스트로 담으면 파라미터가 등록되지 않습니다. ModuleList 여야 합니다.
        #    매번 새로 만드는 것이라 층마다 가중치가 서로 다릅니다('다르게 두 번 읽기').
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, src, mask):
        # √d_model 을 곱하는 이유: 위치인코딩의 진폭은 항상 ~1인데 임베딩은 처음엔 값이 작습니다.
        # 곱하지 않으면 단어 뜻이 위치정보에 묻힙니다(원논문 관례).
        x = self.pos(self.emb(src) * math.sqrt(self.d_model))
        for l in self.layers:                       # 앞 층의 출력이 뒤 층의 입력
            x = l(x, mask)
        return x                                    # 디코더 크로스어텐션의 K·V 로 재사용됩니다


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑧: 디코더 (노트북 셀 29)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/decoder.html
# ─────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """초안을 쓰고 또 쓰며, 매번 질문을 곁눈질해 답의 표현을 겹겹이 다듬습니다.

    Encoder 와 뼈대가 같고, 매 층에 enc·tgt_mask·src_mask 를 함께 넘기는 점만 다릅니다.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        # 답 단어장 기준 임베딩. 질문 쪽 임베딩과는 크기부터 다른, 완전히 별개의 표입니다.
        # 나중에 부품⑨에서 이 표를 출력층과 공유합니다(weight tying).
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, tgt, enc, tgt_mask, src_mask):
        x = self.pos(self.emb(tgt) * math.sqrt(self.d_model))
        # enc 는 매 층에 '똑같은 값'이 그대로 들어갑니다(읽기 전용 참고자료).
        for l in self.layers:
            x = l(x, enc, tgt_mask, src_mask)
        # 출력 마지막 차원은 답 단어장 크기가 아니라 d_model 입니다. 단어 점수로 바꾸는 일은 출력층이 맡습니다.
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑨: 전체 조립 (노트북 셀 31)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/transformer-assembly.html
# ─────────────────────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    """듣기 통역사(인코더)·말하기 통역사(디코더)·마이크(출력층)를 한 부스에 앉히는 조립 도면.

    하이퍼파라미터 기본값을 정하는 곳은 이 클래스뿐입니다. 다른 부품은 인자로 받아 쓰기만 합니다.
    우리 예제 전체 파라미터 235,520개 = 인코더 100,864 + 디코더 134,656.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4,
                 num_layers=2, d_ff=256, dropout=0.1, max_len=32):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)
        # bias=False 인 이유는 아래 가중치 공유 때문입니다. nn.Embedding 은 bias 개념이 없는
        # 순수 조회 테이블이라, 짝을 맞추려면 출력층도 bias 없는 순수 행렬곱이어야 합니다.
        self.out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # weight tying: 복사가 아니라 대입입니다. 두 이름이 '같은 텐서'를 가리킵니다.
        # 같은 텐서라서 named_parameters() 에 out.weight 가 따로 나타나지 않습니다.
        self.out.weight = self.decoder.emb.weight     # weight tying(가중치 공유)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """학습용 문. 정답 전체를 한 번에 넣어(티처 포싱) 모든 위치의 점수를 한 번에 냅니다."""
        return self.out(self.decoder(tgt, self.encoder(src, src_mask), tgt_mask, src_mask))

    def encode(self, src, src_mask):
        """추론용 문(1회). 질문은 답을 만드는 동안 바뀌지 않으므로 딱 한 번만 인코딩해 재사용합니다."""
        return self.encoder(src, src_mask)

    def decode_step(self, tgt, enc, tgt_mask, src_mask):
        """추론용 문(반복). 답이 한 단어 늘어날 때마다 다시 부릅니다. enc 는 인자로 받아 재사용."""
        return self.out(self.decoder(tgt, enc, tgt_mask, src_mask))


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ⑩: 마스크 만들기 (노트북 셀 33)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/masking.html
# ─────────────────────────────────────────────────────────────────────────────
def make_padding_mask(seq, pad_id):
    """길이 맞추려고 넣은 <pad> 자리를 가립니다. (B, S) → (B, 1, 1, S)

    unsqueeze 를 두 번 하는 이유: 어텐션 점수는 (B, 헤드, S_q, S_k) 라서
    헤드축과 Query축 양쪽에 자동으로 늘어나야(브로드캐스팅) 합니다.
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)              # 패딩 위치 가림


def make_causal_mask(seq_len, device):
    """미래 단어를 못 보게 하는 하삼각(causal) 마스크. (1, 1, L, L)

    torch.tril → i번째 단어는 0..i 까지만 볼 수 있습니다. True 개수는 n(n+1)/2.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool().unsqueeze(0).unsqueeze(0)


def make_decoder_mask(tgt, pad_id):
    """디코더 셀프어텐션용 = 패딩 마스크 AND 미래 가림 마스크. (B, 1, T, T)

    ⚠️ 추론(answer)에서는 이 함수를 쓰지 않고 make_causal_mask 만 씁니다.
       한 문장씩 필요한 길이만 키워가므로 애초에 <pad> 가 끼어들 일이 없습니다.
    """
    return make_padding_mask(tgt, pad_id) & make_causal_mask(tgt.size(1), tgt.device)


# ─────────────────────────────────────────────────────────────────────────────
# 4단계: 배치 만들기 (노트북 셀 36)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/training-loop.html
# ─────────────────────────────────────────────────────────────────────────────
def build_batches(pairs, sv, tv):
    """길이가 제각각인 문장들을 <pad> 로 줄 세워 하나의 직사각형 표로 만듭니다.

    질문은 add_special=False, 답은 add_special=True 로 서로 다르게 넣는 것이 핵심입니다.
    질문은 통째로 읽기만 하므로 시작·끝 신호가 필요 없고, 답은 생성 시작/끝 신호가 필요합니다.
    """
    # 질문(src): 특수토큰 없이 인코딩. 실측 첫 질문 → tensor([12, 6, 10, 8, 11])
    src = [torch.tensor(sv.encode(q, add_special=False)) for q, _ in pairs]
    # 답(tgt): <sos>…<eos> 를 붙여 인코딩. 실측 첫 답 → tensor([1, 13, 15, 5, 4, 2])
    tgt = [torch.tensor(tv.encode(a, add_special=True)) for _, a in pairs]
    # 질문은 전부 5단어라 (6, 5) — 채울 게 없어 패딩 0칸입니다.
    src = pad_sequence(src, batch_first=True, padding_value=sv.pad_id)   # 길이 맞추기
    # 답은 가장 긴 것이 6칸이라 (6, 6). 이 0들은 CrossEntropyLoss(ignore_index=pad_id)가 빼 줍니다.
    tgt = pad_sequence(tgt, batch_first=True, padding_value=tv.pad_id)
    return src, tgt


# ─────────────────────────────────────────────────────────────────────────────
# 5단계: 답 생성 함수 (노트북 셀 42)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/greedy-decoding.html
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def answer(model, sv, tv, question, max_len=20):
    """<sos> 부터 한 단어씩 이어 붙이는 끝말잇기(자기회귀 greedy 디코딩).

    질문 인코딩 1회 → (마스크 → 점수 계산 → 1등 선택 → 이어 붙이기) 반복 → <eos> 면 종료.
    반환: (steps, 완성된 답 문자열, cross)
    """
    model.eval()                    # 드롭아웃 끔 → 결과가 결정적
    # 질문을 번호 텐서로. [ ] 로 감싸 배치축을 만들어 (1, S_src) 로.
    src = torch.tensor([sv.encode(question, add_special=False)])
    src_mask = make_padding_mask(src, sv.pad_id)
    # 질문 인코딩은 한 번만 하고 이후 스텝에서 재사용합니다.
    enc = model.encode(src, src_mask)
    # gen: 지금까지 생성한 답 번호들(<sos> 로 시작). steps: (지금까지답, 다음단어) 기록.
    gen, steps = [tv.sos_id], []
    for _ in range(max_len):
        tgt = torch.tensor([gen])                              # (1, 현재길이)
        # 추론 땐 패딩이 없으니 인과 마스크만 있으면 됩니다(학습과 같은 함수를 쓰려는 것).
        tgt_mask = make_causal_mask(tgt.size(1), tgt.device)
        # 매 스텝 '지금까지의 답 전체'를 처음부터 다시 통과시킵니다(KV 캐시 없는 교육용 단순 구현).
        logits = model.decode_step(tgt, enc, tgt_mask, src_mask)
        # 마지막 칸이 곧 '방금 자란 자리'입니다. 앞쪽 칸들은 이미 확정된 단어의 점수입니다.
        nxt = logits[0, -1].argmax().item()               # 마지막 위치의 최고점 단어
        steps.append((tv.decode(gen), tv.itos[nxt]))
        gen.append(nxt)                                   # 다음 스텝 입력이 됨
        if nxt == tv.eos_id:                              # 문장이 끝났으므로 생성 중단
            break
    # 완성된 답 전체를 한 번 더 통과시켜 히트맵 세로축(<eos> 포함 전체 길이)을 완성합니다.
    tgt = torch.tensor([gen]); tgt_mask = make_causal_mask(tgt.size(1), tgt.device)
    model.decode_step(tgt, enc, tgt_mask, src_mask)       # 어텐션 확보용 한 번 더
    # 마지막 디코더 층의 크로스어텐션 가중치를 꺼내 헤드 평균을 냅니다. (답길이, 질문길이)
    cross = model.decoder.layers[-1].cross_attn.last_attn_weights[0].mean(0)
    return steps, tv.decode(gen), cross


# ─────────────────────────────────────────────────────────────────────────────
# 6단계: 어텐션 히트맵 (노트북 셀 45)
# ─────────────────────────────────────────────────────────────────────────────
def show_attention(question, answer_text, cross, save_path=None):
    """답의 각 단어가 질문의 어느 단어에 주목했는지 색으로 보여 줍니다(밝을수록 크게 주목).

    save_path 를 주면 창을 띄우는 대신 그 경로에 PNG 로 저장합니다(노트북에서는 생략).
    """
    # 축 눈금에 쓸 질문/답 토큰 리스트.
    qs, ans_toks = tokenize(question), tokenize(answer_text)
    # cross(답길이×질문길이)에서 실제 토큰 개수만큼 잘라 numpy 배열로. (특수토큰 여백 제거)
    w = cross[:len(ans_toks), :len(qs)].numpy()
    # 토큰 수에 비례해 그림 크기를 잡아 글자가 겹치지 않게 합니다.
    plt.figure(figsize=(1.1 * len(qs) + 1, 0.7 * len(ans_toks) + 1))
    # viridis = 낮음(보라) ~ 높음(노랑) 컬러맵.
    plt.imshow(w, aspect="auto", cmap="viridis")
    plt.xticks(range(len(qs)), qs, rotation=30, ha="right")
    plt.yticks(range(len(ans_toks)), ans_toks)
    plt.xlabel("질문 토큰"); plt.ylabel("생성한 답 토큰")
    plt.title(f"'{answer_text}' 이(가) 주목한 곳")
    plt.colorbar(); plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=150)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 부품 체크포인트 (노트북 셀 10·16·19·34) — `python mini_transformer.py` 로 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 사전: 특수토큰 번호가 기대대로인지 확인(<pad>=0, <sos>=1).
    _sv, _tv = Vocab([q for q, _ in DATA]), Vocab([a for _, a in DATA])
    assert _sv.pad_id == 0 and _tv.sos_id == 1
    print("✅ 사전 OK")

    # 어텐션 공식: Q·K 점수가 모두 같으면(여기선 전부 1) 두 Key 에 0.5씩 균등 분배돼야 함.
    _c, _w = scaled_dot_product_attention(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4),
                                          torch.arange(8.).reshape(1, 1, 2, 4))
    assert torch.allclose(_w, torch.full((1, 1, 2, 2), 0.5))   # 점수가 같으면 0.5씩 균등
    print("✅ 어텐션 공식 OK")

    # 멀티헤드: 입력 shape (1,3,8)을 그대로 (1,3,8)로 돌려주는지(차원 보존) 확인.
    _m = MultiHeadAttention(8, 4, 0.0)
    assert _m(torch.randn(1, 3, 8), torch.randn(1, 3, 8), torch.randn(1, 3, 8)).shape == (1, 3, 8)
    print("✅ 멀티헤드 OK")

    # 마스크: 3x3 인과 마스크의 True 개수가 1+2+3=6인지 확인(하삼각형이 제대로 만들어졌는지).
    assert make_causal_mask(3, torch.device("cpu")).sum().item() == 6   # 1+2+3
    print("✅ 마스크 OK")
