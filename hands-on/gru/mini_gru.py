# -*- coding: utf-8 -*-
"""미니 GRU 공통 모듈 — 미니 트랜스포머 실습의 부품 구성을 그대로 GRU 로 옮긴 것.

🖼️ 그림(회의록 한 장): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-01-hidden-note.webp

[한 줄 요약] 회의록 한 장을 계속 고쳐 쓰는 서기입니다. 발언(단어)이 들어올 때마다 새 종이를
꺼내지 않고, 있던 그 한 장을 **얼마나 고칠지**만 정합니다.

하는 일
    0단계 준비            → setup_korean_font()
    1단계 데이터           → DATA (미니 트랜스포머와 글자 하나까지 같은 6쌍)
    2단계 토큰화/단어장    → PAD/SOS/EOS/UNK, SPECIALS, tokenize(), Vocab
    3단계 부품 ①~④        → MyGRUCell · GRUEncoder · GRUDecoder · GRUSeq2Seq
    4단계 배치 만들기      → build_batches()
    5단계 답 생성          → answer()
    6단계 게이트 관찰      → show_gates() · encode_trace()

⭐ 0~2단계와 4단계는 `mini-transformer/mini_transformer.py` 와 **글자 하나까지 같습니다(의도적 복제)**.
   왜 import 하지 않고 복제했는가 — 폴더명 `mini-transformer` 에 하이픈이 있어 파이썬 모듈로
   불러올 수 없고, `sys.path` 를 손대는 코드는 입문자에게 본 내용보다 어렵기 때문입니다.
   또 이 폴더만 Colab 에 올려도 혼자서 돌아가야 합니다.
   ⚠️ DATA 나 Vocab 을 고칠 일이 생기면 **두 폴더를 함께** 고쳐야 합니다. 한쪽만 고치면
      "같은 데이터로 두 구조를 비교한다"는 이 실습의 전제가 깨집니다.

⭐ 코드가 같다는 사실 자체가 메시지입니다 — 데이터를 다루는 방법은 아키텍처와 무관합니다.
   정말로 달라지는 곳은 3단계 부품 하나(어텐션 → 게이트)뿐입니다.

⭐ 이 모듈은 부품만 제공합니다. 학습은 `training.py`, 추론은 `reasoning.py` 가 합니다.

실행:  python mini_gru.py      (부품 셀프테스트만 돌립니다. 학습은 하지 않습니다)
"""
import logging
import math
import sys

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# ⚠️ matplotlib 을 여기서 import 하지 않는 이유: 그림은 6단계에서만 필요한데, 여기서 불러오면
#    matplotlib 이 없는 환경에서 **부품 셀프테스트와 학습까지 함께 막힙니다**.
#    그림 그리는 함수 안에서만 늦게 불러(lazy import) 없으면 그림만 건너뛰게 만듭니다.

# Windows 기본 콘솔(cp949)은 '✅' 같은 기호를 인코딩하지 못해 UnicodeEncodeError 로 죽습니다.
# 노트북(Colab/Jupyter)은 UTF-8 이라 문제가 없지만, 스크립트로 돌릴 때를 대비해 맞춰 둡니다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 0단계: 그래프 한글 폰트 준비
# ─────────────────────────────────────────────────────────────────────────────
def setup_korean_font():
    """히트맵 축에 한글이 깨지지 않도록 폰트를 잡아 줍니다.

    Colab 이면 나눔고딕(`!apt-get install fonts-nanum` 으로 설치된 것)을 직접 등록하고,
    로컬이면 OS 에 흔한 한글 폰트를 순서대로 탐색합니다.
    matplotlib 이 없으면 조용히 넘어갑니다(그림만 못 그리고 나머지는 다 돌아갑니다).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return                                                        # 그림 없이도 실습은 계속됨
    logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)  # 사소한 폰트 경고 숨김
    plt.rcParams["axes.unicode_minus"] = False                        # 마이너스 기호 깨짐 방지
    try:
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        plt.rcParams["font.family"] = "NanumGothic"                   # Colab: 나눔고딕
    except Exception:
        for _n in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:    # 로컬 폴백
            if _n in {f.name for f in font_manager.fontManager.ttflist}:
                plt.rcParams["font.family"] = _n
                break


# ─────────────────────────────────────────────────────────────────────────────
# 1단계: 아주 작은 '질문→답' 데이터 (mini_transformer.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────
# 문장 '틀'은 똑같이 두고 '대상' 단어(먹구름/별/해/…)만 바꿨습니다.
# 답을 가르는 단서가 오직 그 한 단어뿐이라, 모델이 반드시 거기에 '주목'하게 됩니다.
# ⭐ GRU 에게는 이 설정이 특히 잔인합니다 — 어텐션이 없으니 '주목'할 방법이 없고,
#    질문을 읽어 가는 동안 회의록 한 장을 그 단어에서 크게 고쳐 두는 수밖에 없습니다.
DATA = [
    ("하늘에 먹구름이 보이면 뭐가 생각나", "비가 올 것 같아"),   # 핵심 쌍: '먹구름' → '비'
    ("하늘에 별이 보이면 뭐가 생각나", "밤이 깊었나 봐"),       # '별' → '밤'
    ("하늘에 해가 보이면 뭐가 생각나", "아침이 밝았구나"),       # '해' → '아침'
    ("하늘에 무지개가 보이면 뭐가 생각나", "비가 그쳤나 봐"),     # '무지개' → '비 그침'
    ("하늘에 눈송이가 보이면 뭐가 생각나", "겨울이 왔구나"),      # '눈송이' → '겨울'
    ("하늘에 노을이 보이면 뭐가 생각나", "저녁이 되었네"),        # '노을' → '저녁'
]


# ─────────────────────────────────────────────────────────────────────────────
# 2단계: 토큰화 & 단어장(Vocab) — mini_transformer.py 와 동일
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
# 3단계 부품 ①: GRU 셀 — 게이트 두 개로 회의록 한 장을 고쳐 쓰기
# 🖼️ 그림(수도 레버):   https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-02-update-gate.webp
# 🖼️ 그림(덮어 두기):   https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-03-reset-gate.webp
# 🖼️ 그림(셀 4줄 흐름): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-04-cell-flow.webp
# ─────────────────────────────────────────────────────────────────────────────
class MyGRUCell(nn.Module):
    """단어 하나를 읽고 '기억 메모(은닉 상태)' 한 장을 고쳐 쓰는 한 칸.

    비유 — 회의록을 맡은 서기입니다. 새 발언(x)을 들을 때마다 이렇게 정합니다.
      · 리셋 게이트 r : "초안을 쓸 때 지난 회의록을 얼마나 참고할까?" (0 = 아예 안 봄)
      · 갱신 게이트 z : "완성본에서 회의록을 얼마나 그대로 둘까?"   (1 = 한 글자도 안 고침)
      · 후보     n : r 로 걸러 본 과거 + 새 발언으로 만든 '고쳐 쓸 초안'
    최종 = (1-z)·초안 + z·기존. 두 비율의 합이 항상 1이라 값이 발산하지 않습니다.

    ⭐ 트랜스포머의 어텐션이 "누구를 볼지 고르는" 부품이었다면, 게이트는 "얼마나 고칠지 정하는"
       부품입니다. 고를 대상이 없습니다 — 볼 수 있는 것은 직전 회의록 한 장뿐이니까요.

    학습되는 값: Linear 두 개(x2h, h2h)뿐입니다. sigmoid·tanh 에는 배울 것이 없습니다.
    입출력: x = (배치, 입력칸수), h = (배치, 은닉칸수) → h_new = (배치, 은닉칸수)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        # 선반 하나를 세 칸(r·z·n)으로 나눠 씁니다. Linear 6개를 따로 두는 것과 값은 완전히 같고,
        # 행렬곱이 2번으로 줄어 빠릅니다. 출력 칸수가 3배(192)인 이유가 바로 이것입니다.
        # ⚠️ 아래 두 줄의 '선언 순서'가 재현성을 정합니다. 순서를 바꾸면 난수를 꺼내 쓰는 순서가
        #    달라져 **초기 가중치 전체가 달라집니다**. 시드를 42로 고정해도 결과가 안 맞게 됩니다.
        self.x2h = nn.Linear(input_size, 3 * hidden_size)     # 새 입력 → 세 칸
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)    # 기존 기억 → 세 칸

    def forward(self, x, h):
        """한 스텝 갱신. 반환: (새 기억, 리셋 게이트, 갱신 게이트)

        게이트 2개를 함께 돌려주는 이유: 6단계에서 이 값을 히트맵으로 그려 '모델의 속'을
        들여다봅니다. `nn.GRU` 를 쓰면 게이트를 꺼낼 수 없어 그림을 그릴 수 없습니다.
        (트랜스포머는 MultiHeadAttention.last_attn_weights 에 보관했지만, 게이트는 스텝마다
         값이 바뀌므로 속성에 담으면 덮어써집니다. 그래서 반환값으로 넘깁니다.)
        """
        # chunk(3) = 192칸을 64칸씩 세 토막으로 자르기.
        # ⚠️ 자르는 순서 (r, z, n) 은 PyTorch 규약입니다. (z, r, n) 등으로 바꾸면 학습은 되지만
        #    아래 셀프테스트의 `nn.GRUCell 일치` 검사에서 걸립니다.
        xr, xz, xn = self.x2h(x).chunk(3, dim=-1)
        hr, hz, hn = self.h2h(h).chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)          # 리셋 게이트: 0~1 비율
        z = torch.sigmoid(xz + hz)          # 갱신 게이트: 0~1 비율
        # ⚠️ r 은 은닉 상태에 바로 곱하는 게 아니라 '변환한 뒤(hn)' 에 곱합니다.
        #    맞음:  n = tanh(xn + r * hn)
        #    틀림:  n = tanh(xn + self.h2h(r * h)) 의 n 토막
        #    두 줄 다 에러 없이 돌고 학습도 되지만, PyTorch nn.GRUCell 과 값이 달라집니다.
        #    (교과서가 tanh(Wx·x + Wh·(r⊙h)) 로 적어 두어 헷갈리기 쉬운 지점입니다)
        n = torch.tanh(xn + r * hn)         # 후보(고쳐 쓸 초안): -1~1 값
        # ⚠️ z 는 '새것을 받을 비율'이 아니라 '옛 기억을 지킬 비율'입니다(PyTorch 관례).
        #    이름이 '갱신 게이트'라서 반대로 외우기 쉽습니다. 뒤집어도 학습은 되지만
        #    게이트 히트맵의 해석이 완전히 반대가 되어, 실습의 결론이 거짓이 됩니다.
        # ⚠️ 게이트(r, z)는 반드시 sigmoid(0~1, '비율')이고 후보(n)는 tanh(-1~1, '값')입니다.
        #    게이트에 tanh 를 쓰면 '-30% 를 기억한다' 같은 말이 되어 학습이 요동칩니다.
        return (1 - z) * n + z * h, r, z    # 두 비율의 합 = 1 → |h| 가 1을 절대 넘지 않음


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ②: 인코더 — 질문을 한 단어씩 읽어 '요약 벡터' 한 개로 압축
# 🖼️ 그림(순차 읽기): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-05-encoder-loop.webp
# ─────────────────────────────────────────────────────────────────────────────
class GRUEncoder(nn.Module):
    """질문 5단어를 **차례대로** 읽어, 마지막에 남은 회의록 한 장을 요약(context)으로 넘깁니다.

    ⭐ 트랜스포머 인코더와 결정적으로 다른 점: `for` 루프입니다.
       트랜스포머는 5단어를 한 번의 행렬 연산으로 동시에 처리했습니다(그래서 GPU 로 병렬화됨).
       GRU 는 앞 단어를 처리한 결과(h)가 있어야 다음 단어를 처리할 수 있습니다.
       "다트를 한 발씩 던진다" — 실력이 늘는 속도는 비슷한데 던지는 시간이 길어집니다.

    ⭐ 사라진 부품: PositionalEncoding 이 없습니다.
       순서대로 한 개씩 넣으므로 순서 정보가 **구조에 이미 들어 있습니다**. 트랜스포머 실습에서
       가장 어려웠던 부품이 여기서는 아예 필요가 없어졌습니다 — 왜 그 부품이 있었는지가
       거꾸로 보이는 순간입니다.
    """

    def __init__(self, vocab_size, emb_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_dim)   # 번호 → 뜻벡터. (14, 64) 표
        self.drop = nn.Dropout(dropout)                # 고정 무늬에만 매달리지 않게 일부를 지움
        self.cell = MyGRUCell(emb_dim, hidden_size)    # 부품 ① 을 시간축으로 반복 사용

    def forward(self, src_ids):
        """반환: (요약 벡터, 시점별 은닉, 시점별 리셋, 시점별 갱신)

        src_ids = (배치, 질문길이). 실측 shape: (6, 5)
        요약 벡터 = (배치, 64) / 시점별 값 = (배치, 5, 64)
        """
        b, s = src_ids.shape
        e = self.drop(self.emb(src_ids))                          # (6, 5, 64) 뜻벡터
        # ⚠️ 문장을 새로 읽을 때마다 회의록을 백지(0)로 다시 만들어야 합니다. 앞 문장의 내용을
        #    지우지 않으면 지금 질문과 상관없는 옛 기억이 답에 섞입니다.
        # ⚠️ (배치, 64) 가 아니라 (64,) 로 만들면 에러 없이 브로드캐스팅되어 6문장이
        #    '같은 회의록 한 장'을 나눠 쓰게 됩니다 — 조용히 틀리는 가장 무서운 실수입니다.
        h = torch.zeros(b, self.hidden_size, device=src_ids.device, dtype=e.dtype)
        hs, rs, zs = [], [], []                                   # 시점별 기록(그림용)
        for t in range(s):                                        # ⭐ 여기가 '한 단어씩'입니다
            # ⚠️ 우리 텐서는 (배치, 시간, 칸) 순서입니다. t 번째 단어는 e[:, t] 이지 e[t] 가
            #    아닙니다. e[t] 를 쓰면 't번 문장을 통째로' 한 스텝에 넣는 셈이 되어,
            #    에러 없이 shape 만 맞고 의미는 완전히 망가집니다.
            h, r, z = self.cell(e[:, t], h)                       # 회의록을 한 번 고쳐 씀
            hs.append(h); rs.append(r); zs.append(z)
        # stack(dim=1) → 시간축을 가운데에 끼워 (배치, 시간, 칸) 순서를 유지합니다.
        # ⭐ 마지막 h 하나만 디코더로 넘어갑니다. 질문 5단어가 64칸 한 장에 눌려 담기는 것 —
        #    이것이 뒤에서 다룰 '정보 병목'의 정체입니다.
        return h, torch.stack(hs, dim=1), torch.stack(rs, dim=1), torch.stack(zs, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ③: 디코더 — 요약 벡터 한 개만 들고 답을 한 단어씩 쓰기
# 🖼️ 그림(배턴 하나): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-06-decoder-baton.webp
# ─────────────────────────────────────────────────────────────────────────────
class GRUDecoder(nn.Module):
    """이어달리기의 두 번째 주자입니다. 질문지는 이미 손에서 떠났고, 앞 주자가 넘긴 배턴
    (요약 벡터 64칸)만 들고 답을 씁니다.

    ⭐ 트랜스포머의 디코더는 "답안지 뒷장은 못 보고, 문제지는 계속 곁눈질하는 수험생"이었습니다.
       GRU 디코더는 **곁눈질을 못 하게 만든 버전**입니다. 그 곁눈질이 크로스어텐션이었고,
       여기엔 그 부품이 없습니다.

    요약 벡터를 두 경로로 받습니다.
      ① 출발점(h0) — 배턴을 손에 쥔 상태로 출발합니다
      ② 매 스텝 입력에 이어 붙이기 — 답이 길어져도 배턴을 계속 참고합니다
    ⭐ 두 경로를 다 열어 줘도 **결국 그 64칸이 전부**입니다. 질문 원문은 어디에도 없습니다.
       이것이 병목이라는 말의 뜻입니다.

    ⭐ 사라진 부품: 마스크 3종(make_padding_mask / make_causal_mask / make_decoder_mask)이 없습니다.
       미래 단어를 가릴 필요가 없습니다 — **아직 계산조차 안 했으므로 볼 수가 없습니다.**
       트랜스포머는 전 단어를 동시에 계산했기에 일부러 가려야 했던 것입니다.
    """

    def __init__(self, vocab_size, emb_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_dim)   # 답 단어장용. (18, 64) 표
        self.drop = nn.Dropout(dropout)
        # 입력 칸수가 emb_dim + hidden_size = 128 인 이유: [답 임베딩 64 ; 요약 벡터 64] 를
        # 이어 붙여 한꺼번에 넣기 때문입니다. 그래서 디코더 x2h 만 파라미터가 2배입니다.
        self.cell = MyGRUCell(emb_dim + hidden_size, hidden_size)

    def forward(self, dec_in_ids, context):
        """학습용 문. 정답 전체를 한 번에 넣어(티처 포싱) 모든 시점의 은닉을 냅니다.

        dec_in_ids = (배치, 답길이-1). 실측 shape: (6, 5)
        context    = (배치, 64) 인코더가 넘긴 요약 벡터
        반환       = (배치, 답길이-1, 64) 시점별 은닉
        """
        b, t_len = dec_in_ids.shape
        e = self.drop(self.emb(dec_in_ids))                       # (6, 5, 64)
        h = context                                               # ① 배턴을 쥐고 출발
        outs = []
        for t in range(t_len):
            # ② 매 스텝 [이번 답 단어 ; 요약 벡터] 를 이어 붙여 넣습니다. (6, 128)
            x = torch.cat([e[:, t], context], dim=-1)
            h, _, _ = self.cell(x, h)                             # 게이트는 학습 중엔 안 씀
            outs.append(h)
        # ⚠️ 인코더가 돌려주는 것은 '마지막 은닉 1개', 디코더가 돌려주는 것은 '모든 시점'입니다.
        #    인코더에서도 전체 시점을 모아 디코더에 넘기고 싶어지면 잠깐 멈추세요 —
        #    그것을 하는 순간 여러분은 어텐션을 다시 발명하고 있는 것입니다(그게 트랜스포머입니다).
        return torch.stack(outs, dim=1)                           # (6, 5, 64)

    def step(self, prev_id, h, context):
        """추론용 문(반복). 방금 만든 단어 하나만 넣어 회의록을 한 번 고쳐 씁니다.

        prev_id = (배치,) 직전에 고른 단어 번호 / h = (배치, 64) 지금까지의 회의록
        ⭐ 트랜스포머의 answer() 는 매 스텝 '지금까지의 답 전체'를 처음부터 다시 통과시켰습니다.
           GRU 는 회의록 한 장만 넘겨받으면 되므로 **직전 단어 하나**로 충분합니다.
           같은 일을 훨씬 적은 계산으로 하는 것 — GRU 가 이기는 유일한 지점입니다.
        """
        e = self.emb(prev_id)                                     # (배치, 64)
        x = torch.cat([e, context], dim=-1)                       # (배치, 128)
        return self.cell(x, h)                                    # (새 h, r, z)


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 부품 ④: 조립 — 인코더 + 디코더 + 출력층
# 🖼️ 그림(트랜스포머 대비): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-08-vs-transformer.webp
# ─────────────────────────────────────────────────────────────────────────────
class GRUSeq2Seq(nn.Module):
    """질문 → 요약 벡터 → 답. 부품 ①②③을 이어 붙이고 마지막에 '단어 점수'로 바꿉니다.

    ⭐ 하이퍼파라미터 기본값을 정하는 곳은 이 클래스뿐입니다. 다른 부품은 인자로 받아 쓰기만 합니다.
       값을 트랜스포머와 똑같이 맞춰 두었습니다(emb 64 / hidden 64 / dropout 0.1).
       구조만 바꾸고 나머지를 전부 고정해야, 결과 차이가 100% '구조 때문'이 됩니다.

    ⭐ 사라진 부품이 또 있습니다.
       · PositionwiseFeedForward — 게이트 계산 자체가 비선형 변환 역할을 겸합니다.
       · EncoderLayer / DecoderLayer (LayerNorm·잔차 연결) — 층이 1개라 쌓을 게 없고,
         게이트가 값의 크기를 스스로 억제해 줍니다(|h| ≤ 1).
       트랜스포머는 부품이 ①~⑩ 열 개였습니다. 여기는 ①~④ 네 개입니다.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 emb_dim=64, hidden_size=64, dropout=0.1):
        super().__init__()
        # ⚠️ 아래 세 줄도 선언 순서가 재현성을 정합니다(난수 소비 순서). 순서 변경 금지.
        self.encoder = GRUEncoder(src_vocab_size, emb_dim, hidden_size, dropout)
        self.decoder = GRUDecoder(tgt_vocab_size, emb_dim, hidden_size, dropout)
        self.drop = nn.Dropout(dropout)                                   # 출력층 직전
        # 은닉 64칸 → 답 단어 18개의 점수. bias=False 는 아래 가중치 공유의 조건입니다.
        self.out = nn.Linear(hidden_size, tgt_vocab_size, bias=False)
        # 가중치 공유(weight tying): '단어→뜻벡터' 표와 '뜻벡터→단어점수' 표를 **같은 표**로 씁니다.
        # 트랜스포머 실습과 같은 기법이며, RNN 에도 그대로 통합니다.
        # 실측(seed 42, 400에폭): 공유하면 파라미터가 65,408 → 64,256 으로 1,152개 줄어드는데
        #   평가 loss 는 0.0686 → 0.0025 로 **27배 좋아집니다**. 작아졌는데 더 잘하는 것 —
        #   같은 표를 두 번 쓰니 그 표가 두 배로 다듬어지기 때문입니다.
        # ⚠️ `.data = ` 로 값을 복사하면 두 표가 따로 학습되어 공유가 아닙니다. 아래처럼
        #    **같은 객체를 대입**해야 합니다(hidden_size == emb_dim 일 때만 모양이 맞습니다).
        self.out.weight = self.decoder.emb.weight

    def forward(self, src_ids, dec_in_ids):
        """학습용 문. 반환: (배치, 답길이-1, 답단어수) 점수. 실측 shape: (6, 5, 18)"""
        context, _, _, _ = self.encoder(src_ids)                  # 질문 → 요약 벡터
        h_all = self.decoder(dec_in_ids, context)                 # 요약 + 정답 → 시점별 은닉
        return self.out(self.drop(h_all))                         # 은닉 → 단어 점수

    def encode(self, src_ids):
        """추론용 문(1회). 질문은 답을 만드는 동안 바뀌지 않으므로 딱 한 번만 읽습니다."""
        return self.encoder(src_ids)                              # (context, hs, rs, zs)

    def decode_step(self, prev_id, h, context):
        """추론용 문(반복). 반환: (단어 점수, 새 h, 리셋, 갱신)"""
        h, r, z = self.decoder.step(prev_id, h, context)
        return self.out(h), h, r, z


# ─────────────────────────────────────────────────────────────────────────────
# 4단계: 배치 만들기 — mini_transformer.py 와 동일
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
# 5단계: 답 생성 함수 (greedy 디코딩)
# 📖 https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/greedy-decoding.html
# ─────────────────────────────────────────────────────────────────────────────
def encode_question(sv, question, warn=True):
    """질문 문장 → (1, 길이) 번호 텐서. 빈 문장과 모르는 단어를 여기서 걸러 안내합니다.

    ⚠️ 빈 문장 가드가 없으면 GRU 는 회의록 백지(h=0) 그대로 답을 만들어 **조용히 틀린 결과**를
       냅니다. 에러도 안 나므로 학습자가 원인을 찾을 수 없습니다. 그래서 막아 둡니다.
    """
    toks = tokenize(question)
    if not toks:
        raise ValueError(
            "빈 문장은 처리할 수 없습니다. 단어를 하나 이상 입력하세요. "
            '(예: "하늘에 먹구름이 보이면 뭐가 생각나")'
        )
    unknown = [t for t in toks if t not in sv.stoi]
    if unknown and warn:
        print(f"⚠️ 단어장에 없는 단어 {len(unknown)}개를 <unk> 로 처리했습니다: {unknown}")
        print("   학습한 6문장에 없는 단어라 답이 엉뚱할 수 있습니다.")
    return torch.tensor([sv.encode(question, add_special=False)])   # (1, 길이)


@torch.no_grad()
def answer(model, sv, tv, question, max_len=20, warn=True):
    """<sos> 부터 한 단어씩 이어 붙이는 끝말잇기(자기회귀 greedy 디코딩).

    질문 읽기 1회(요약 벡터 확보) → (직전 단어 넣기 → 1등 고르기 → 이어 붙이기) 반복 → <eos> 면 종료.
    반환: (steps, 완성된 답 문자열, 질문 시점별 은닉, 질문 시점별 리셋, 질문 시점별 갱신)

    ⚠️ model.eval() 을 빼면 드롭아웃이 켜진 채로 답을 만들어 같은 질문에 매번 다른 답이 나옵니다.
       게이트 값도 흔들려 히트맵을 신뢰할 수 없게 됩니다. @torch.no_grad() 도 함께 챙깁니다.
    """
    model.eval()                                        # 드롭아웃 끔 → 결과가 결정적
    src = encode_question(sv, question, warn=warn)
    # 질문 읽기는 한 번만. 요약 벡터(context)와 관찰용 기록(hs/rs/zs)을 함께 받습니다.
    context, hs, rs, zs = model.encode(src)
    # gen: 지금까지 생성한 답 번호들(<sos> 로 시작). steps: (지금까지답, 다음단어) 기록.
    gen, steps = [tv.sos_id], []
    h = context                                         # 디코더도 배턴을 쥐고 출발
    for _ in range(max_len):
        prev = torch.tensor([gen[-1]])                  # ⭐ 직전 단어 하나만 넣으면 충분
        logits, h, _, _ = model.decode_step(prev, h, context)
        nxt = logits[0].argmax().item()                 # 점수 1등 단어 = greedy
        steps.append((tv.decode(gen), tv.itos[nxt]))
        gen.append(nxt)                                 # 다음 스텝 입력이 됨
        if nxt == tv.eos_id:                            # 문장이 끝났으므로 생성 중단
            break
    return steps, tv.decode(gen), hs, rs, zs


@torch.no_grad()
def encode_trace(model, sv, question, warn=False):
    """질문을 읽는 동안의 은닉 궤적만 뽑습니다. (1, 질문길이, 64)

    두 질문의 궤적을 빼 보면 "어느 단어에서 기억이 갈라졌나"를 숫자로 알 수 있습니다.
    ⭐ 어텐션 히트맵이 없는 GRU 에서 '주목한 곳'을 검산하는 방법입니다.
       문장 틀이 같으면 첫 토큰('하늘에')까지의 은닉은 **비트 단위로 똑같아야** 하고,
       대상 단어 자리에서 비로소 갈라져야 합니다.
    """
    model.eval()
    _, hs, _, _ = model.encode(encode_question(sv, question, warn=warn))
    return hs


# ─────────────────────────────────────────────────────────────────────────────
# 6단계: 게이트 히트맵 — 어텐션 대신 '게이트'를 들여다보기
# ─────────────────────────────────────────────────────────────────────────────
def show_gates(question, resets, updates, save_path=None):
    """질문을 읽는 동안 두 게이트가 어떤 값을 냈는지 색으로 보여 줍니다.

    세로축 = 질문 토큰 5개, 가로축 = 은닉 64칸. 위 그림이 리셋 r, 아래 그림이 갱신 z 입니다.
    ⭐ vmin=0, vmax=1 을 고정합니다. 게이트는 시그모이드 출력이라 물리적 상한이 0/1 이므로,
       자동 스케일을 쓰면 '밝다'의 뜻이 그림마다 달라져 오독을 유발합니다.

    save_path 를 주면 창을 띄우는 대신 그 경로에 PNG 로 저장합니다.
    matplotlib 이 없으면 안내만 하고 넘어갑니다(텍스트 검산으로 결론은 그대로 확인 가능).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 이 없어 히트맵을 건너뜁니다: pip install matplotlib")
        return False
    qs = tokenize(question)
    # (1, 5, 64) 에서 배치축을 떼고 실제 토큰 수만큼 자릅니다.
    r = resets[0, :len(qs)].numpy()
    z = updates[0, :len(qs)].numpy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 1.0 * len(qs) + 2.2), sharex=True)
    for ax, w, name, hint in (
        (axes[0], r, "리셋 게이트 r", "밝을수록 1 (초안 쓸 때 과거를 많이 참고)"),
        (axes[1], z, "갱신 게이트 z", "밝을수록 1 (회의록을 그대로 유지)"),
    ):
        im = ax.imshow(w, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(qs))); ax.set_yticklabels(qs)
        ax.set_title(f"{name} — {hint}", fontsize=11)
        fig.colorbar(im, ax=ax)
    axes[1].set_xlabel("은닉 상태 64칸")
    fig.suptitle(f"'{question}' 를 읽는 동안의 게이트", fontsize=12)
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 부품 체크포인트 — `python mini_gru.py` 로 실행 (학습은 하지 않습니다)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    # ── ① 사전: 특수토큰 번호와 단어장 크기가 기대대로인지 확인 ──────────────────
    _sv, _tv = Vocab([q for q, _ in DATA]), Vocab([a for _, a in DATA])
    assert _sv.pad_id == 0 and _tv.sos_id == 1, "특수토큰 번호가 어긋났습니다. SPECIALS 순서를 확인하세요."
    assert (len(_sv), len(_tv)) == (14, 18), f"단어장 크기가 14/18 이 아닙니다: {len(_sv)}/{len(_tv)}"
    print("✅ 사전 OK                             단어장", len(_sv), "/", len(_tv))

    # ── ② 게이트 범위·shape: 입력을 10배로 키워 일부러 포화시켜도 범위를 지키는지 ──
    _cell = MyGRUCell(5, 7)
    _x, _h = torch.randn(4, 5) * 10.0, torch.randn(4, 7)
    _hn, _r, _z = _cell(_x, _h)
    assert _r.shape == (4, 7) and _z.shape == (4, 7), f"게이트 shape 이상: {tuple(_r.shape)}"
    assert 0.0 <= _r.min().item() and _r.max().item() <= 1.0, "리셋 게이트가 0~1 밖입니다(sigmoid 누락?)"
    assert 0.0 <= _z.min().item() and _z.max().item() <= 1.0, "갱신 게이트가 0~1 밖입니다(sigmoid 누락?)"
    assert _hn.shape == (4, 7), f"은닉 상태 shape 이 (4,7) 이 아닙니다: {tuple(_hn.shape)}"
    print("✅ 게이트 범위·shape OK                r·z 는 0~1, 은닉 칸수는 그대로 보존")

    # ── ③ nn.GRUCell 일치: 우리가 손으로 쓴 4줄이 진짜 GRU 인지 대조 ─────────────
    # PyTorch 의 가중치를 그대로 복사해 넣고 같은 입력을 주면 값이 같아야 합니다.
    # nn.GRUCell 의 weight_ih 는 (3H, in) 이고 위쪽부터 r, z, n 순서로 쌓여 있습니다.
    _ref, _mine = nn.GRUCell(5, 7), MyGRUCell(5, 7)
    with torch.no_grad():
        _mine.x2h.weight.copy_(_ref.weight_ih); _mine.x2h.bias.copy_(_ref.bias_ih)
        _mine.h2h.weight.copy_(_ref.weight_hh); _mine.h2h.bias.copy_(_ref.bias_hh)
    _x1, _h1 = torch.randn(3, 5), torch.randn(3, 7)
    _d1 = (_mine(_x1, _h1)[0] - _ref(_x1, _h1)).abs().max().item()
    assert _d1 < 1e-6, (
        f"직접 구현 셀이 nn.GRUCell 과 다릅니다(최대 오차 {_d1:.2e}). "
        "n 계산에서 r 을 h2h 통과 '후' 에 곱했는지 확인하세요: tanh(xn + r * hn)"
    )
    # 5스텝 이어 돌려 오차가 눈덩이처럼 커지지 않는지도 봅니다.
    # 허용 오차를 1스텝 1e-6 / 5스텝 1e-5 로 다르게 잡는 이유: 두 구현은 수식은 같지만
    # 연산을 묶는 방식이 달라(PyTorch 는 융합 커널) float32 에서 스텝당 ~1e-7 오차가 쌓입니다.
    _a, _b = _h1.clone(), _h1.clone()
    for _ in range(5):
        _xt = torch.randn(3, 5)
        _a, _b = _mine(_xt, _a)[0], _ref(_xt, _b)
    _d5 = (_a - _b).abs().max().item()
    assert _d5 < 1e-5, f"5스텝 누적 후 오차가 큽니다: {_d5:.2e}"
    print(f"✅ nn.GRUCell 일치 OK                  1스텝 {_d1:.2e} / 5스텝 {_d5:.2e}")
    print("   ⭐ 우리가 손으로 쓴 4줄이 진짜 GRU 라는 증거입니다")

    # ── ④ 게이트 방향: z 를 극단으로 밀어 규약이 뒤집히지 않았는지 확인 ───────────
    _fz = MyGRUCell(5, 7)
    with torch.no_grad():
        # z 만 강제로 조작합니다. z 토막은 3H 중 가운데(H ~ 2H) 구간입니다.
        _fz.x2h.weight[7:14].zero_(); _fz.h2h.weight[7:14].zero_()
        _fz.x2h.bias[7:14].fill_(20.0); _fz.h2h.bias[7:14].fill_(20.0)   # sigmoid(40) ≈ 1
    _hp = torch.randn(2, 7)
    assert torch.allclose(_fz(torch.randn(2, 5), _hp)[0], _hp, atol=1e-8), (
        "z≈1 인데 은닉 상태가 바뀌었습니다. h' = (1-z)*n + z*h 의 순서가 뒤집혔는지 확인하세요."
    )
    with torch.no_grad():
        _fz.x2h.bias[7:14].fill_(-20.0); _fz.h2h.bias[7:14].fill_(-20.0)  # sigmoid(-40) ≈ 0
    _x2 = torch.randn(2, 5)
    _hn2, _, _ = _fz(_x2, _hp)
    # z≈0 이면 h' 가 후보 n 과 같아야 합니다. n 을 직접 계산해 비교합니다.
    _xn = _fz.x2h(_x2).chunk(3, dim=-1)[2]
    _hr, _, _hn3 = _fz.h2h(_hp).chunk(3, dim=-1)
    _rr = torch.sigmoid(_fz.x2h(_x2).chunk(3, dim=-1)[0] + _hr)
    assert torch.allclose(_hn2, torch.tanh(_xn + _rr * _hn3), atol=1e-6), \
        "z≈0 인데 h' 가 후보 n 과 다릅니다."
    print("✅ 게이트 방향(z 규약) OK              z≈1 이면 옛 기억 그대로 / z≈0 이면 새 후보 그대로")

    # ── ⑤ 은닉 유계성: 입력을 50배로 키워 20스텝 돌려도 |h| 가 1을 못 넘는지 ────────
    # 근거: h' = (1-z)·n + z·h 는 |n|≤1 과 |h|≤1 의 볼록결합이라 수학적으로 1을 넘을 수 없습니다.
    # ⭐ 그래서 GRU 는 LayerNorm 없이도 값이 터지지 않습니다(트랜스포머는 층마다 필요했습니다).
    _hb = torch.zeros(2, 7)
    for _ in range(20):
        _hb, _, _ = _cell(torch.randn(2, 5) * 50.0, _hb)
    assert _hb.abs().max().item() <= 1.0 + 1e-6, \
        f"은닉 상태가 1을 넘었습니다({_hb.abs().max().item():.4f})."
    print("✅ 은닉 상태 유계성 OK                 입력을 50배로 키워도 |h| ≤ 1")

    # ── ⑥ 순차 처리 인과성: 마지막 토큰만 바꿔도 앞 시점은 그대로여야 함 ──────────
    # 미래 정보가 새고 있으면(전체를 한꺼번에 계산했다면) 앞 시점 값도 함께 변합니다.
    _enc = GRUEncoder(len(_sv), 64, 64, dropout=0.0); _enc.eval()   # 드롭아웃 꺼야 비트 비교 가능
    _ids = torch.tensor([[4, 5, 6, 7, 8]])
    _ids2 = _ids.clone(); _ids2[0, 4] = 9                            # 마지막 토큰만 교체
    with torch.no_grad():
        _, _hs1, _, _ = _enc(_ids)
        _, _hs2, _, _ = _enc(_ids2)
    assert _hs1.shape == (1, 5, 64), f"시점별 은닉 shape 이상: {tuple(_hs1.shape)}"
    assert torch.equal(_hs1[:, :4], _hs2[:, :4]), \
        "마지막 토큰만 바꿨는데 앞쪽 은닉이 변했습니다. 미래 정보가 새고 있습니다."
    assert not torch.equal(_hs1[:, 4], _hs2[:, 4]), \
        "마지막 토큰을 바꿨는데 마지막 은닉이 그대로입니다(입력 미반영)."
    print("✅ 순차 처리 순서 OK                   마지막 토큰만 바꾸면 앞 시점은 그대로")

    # ── ⑦ nn.GRU 전체 시퀀스 일치: 루프 방향과 h0=0 처리가 맞는지 ─────────────────
    _rg, _mc = nn.GRU(5, 7, batch_first=True), MyGRUCell(5, 7)
    with torch.no_grad():
        _mc.x2h.weight.copy_(_rg.weight_ih_l0); _mc.x2h.bias.copy_(_rg.bias_ih_l0)
        _mc.h2h.weight.copy_(_rg.weight_hh_l0); _mc.h2h.bias.copy_(_rg.bias_hh_l0)
    _seq = torch.randn(2, 6, 5)
    with torch.no_grad():
        _ref_out, _ = _rg(_seq)                                      # PyTorch 가 한 번에
        _hm, _outs = torch.zeros(2, 7), []
        for _t in range(6):                                          # 우리는 한 스텝씩
            _hm, _, _ = _mc(_seq[:, _t], _hm)
            _outs.append(_hm)
        _mine_out = torch.stack(_outs, dim=1)
    _dseq = (_mine_out - _ref_out).abs().max().item()
    assert _dseq < 1e-5, f"nn.GRU 전체 시퀀스와 불일치(최대 오차 {_dseq:.2e}). 루프 방향과 h0=0 을 확인하세요."
    print(f"✅ nn.GRU 시퀀스 일치 OK               최대 오차 {_dseq:.2e}")

    # ── 보너스: 조립한 모델의 파라미터 수 (training.py 가 출력할 값과 같아야 함) ───
    _m = GRUSeq2Seq(len(_sv), len(_tv))
    _n = sum(p.numel() for p in _m.parameters())
    _src, _tgt = build_batches(DATA, _sv, _tv)
    assert _src.shape == (6, 5) and _tgt.shape == (6, 6), "배치 shape 이 (6,5)/(6,6) 이 아닙니다."
    assert _m.out.weight is _m.decoder.emb.weight, "가중치 공유가 끊어졌습니다(복사 대신 대입이어야 함)."
    print(f"✅ 조립 OK                             질문 {tuple(_src.shape)} | 답 {tuple(_tgt.shape)} | 파라미터 {_n} 개")
