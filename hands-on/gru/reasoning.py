# -*- coding: utf-8 -*-
"""GRU 추론 수행 — 요약 벡터 한 개로 답을 만들고, 게이트가 한 일을 눈으로 확인합니다.

🖼️ 그림(배턴 하나):  https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-06-decoder-baton.webp
🖼️ 그림(기억 씻김):  https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-07-bottleneck.webp
📖 greedy 디코딩 설명: https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/greedy-decoding.html

[한 줄 요약] 배턴 하나만 받아 들고 뛰는 계주 주자입니다. 질문지는 이미 손에서 떠났고,
남은 것은 앞 주자가 넘겨준 요약 한 장(64칸)뿐입니다.

하는 일
    5단계  질문 하나를 넣어 답을 한 단어씩 만드는 과정을 출력
    6단계  게이트 기록 표 + 히트맵 PNG 저장 + '기억이 갈라진 지점' 검산
    7단계  '대상' 단어만 바꿔 답이 달라지는지 대조 (트랜스포머 실습과 같은 표)
    8단계  기억 씻김 — 질문 뒤에 뜻 없는 단어를 붙이면 답이 무너지는지
    9단계  최근성 편향 — 키워드를 두 개 넣으면 '뒤에 온 것'이 이기는지

⭐ 8·9단계가 이 실습의 결론입니다. 트랜스포머의 크로스어텐션은 질문 어디든 곧바로 다시 볼 수
   있었지만, GRU 디코더는 요약 벡터 한 개만 받습니다. 그 차이가 답을 가릅니다.

`training.py` 가 만든 `mini_gru.pt` 가 먼저 있어야 합니다.

실행:  python training.py     그다음    python reasoning.py
       (PowerShell 에서는 `&&` 를 쓸 수 없어 두 줄로 나눠 실행합니다)
"""
from pathlib import Path

import torch

from mini_gru import (
    DATA,
    GRUSeq2Seq,
    answer,
    encode_trace,
    setup_korean_font,
    show_gates,
    tokenize,
)

# training.py 가 저장한 체크포인트 / 히트맵을 내보낼 경로.
CKPT_PATH = Path(__file__).with_name("mini_gru.pt")
HEATMAP_PATH = Path(__file__).with_name("gate_heatmap.png")

# 5·6단계에서 자세히 들여다볼 대표 질문.
MAIN_QUESTION = "하늘에 먹구름이 보이면 뭐가 생각나"
# 6단계 '기억 갈라짐' 검산용 짝. 문장 틀이 같고 대상 단어만 다릅니다.
DIVERGE_PAIR = ("하늘에 먹구름이 보이면 뭐가 생각나", "하늘에 별이 보이면 뭐가 생각나")
# 8단계: 질문 '뒤'에 붙일 군더더기 반복 횟수. 단어장에 있는 단어라 <unk> 가 아닙니다.
FILLER_WORD, FILLER_COUNTS = "하늘에", [0, 3, 5, 10]
# 9단계: 키워드 두 개를 순서만 바꿔 넣어 봅니다.
ORDER_QUESTIONS = [
    "하늘에 먹구름이 별이 보이면 뭐가 생각나",
    "하늘에 별이 먹구름이 보이면 뭐가 생각나",
]
# 학습한 6개 답을 모아 둔 '집합(set)'. {…} 안에 for 를 쓴 것이 집합 컴프리헨션입니다.
# `for _, a in DATA` 의 `_` 는 "이 값은 안 쓸 거예요" 라는 관례적 이름입니다(질문은 버리고 답만).
# ⭐ 왜 집합인가 — `ans in EXPECTED_ANSWERS` 로 "학습한 답 중 하나인가"만 보면 됩니다.
#    특정 문자열과 정확히 같은지 따지는 것보다 느슨해서, 사소한 차이로 헛되게 실패하지 않습니다.
EXPECTED_ANSWERS = {a for _, a in DATA}


def load_model(ckpt_path=CKPT_PATH):
    """체크포인트에서 모델과 단어장을 되살립니다.

    단어장을 함께 저장해 둔 이유: 번호 체계가 학습 때와 한 칸이라도 어긋나면
    같은 가중치라도 완전히 다른 답이 나옵니다.

    ⭐ 트랜스포머는 여기서 max_len 도 꺼내 모델에 넘겼습니다. GRU 에는 그 키가 없습니다 —
       위치인코딩 표가 없어 문장 길이에 따라 모양이 바뀌는 파라미터가 하나도 없기 때문입니다.
       (그래서 GRU 는 학습 때보다 긴 문장을 넣어도 에러가 나지 않습니다. 8단계에서 확인합니다)
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {ckpt_path}\n먼저 `python training.py` 를 실행하세요."
        )
    # Ctrl+C 로 학습을 끊으면 0바이트 파일이 남을 수 있습니다. 알쏭달쏭한 오류 대신 먼저 안내합니다.
    if ckpt_path.stat().st_size < 1024:
        raise RuntimeError(
            f"체크포인트가 손상되었습니다: {ckpt_path} ({ckpt_path.stat().st_size} 바이트).\n"
            "파일을 삭제하고 `python training.py` 를 다시 실행하세요."
        )
    # weights_only=False: Vocab 객체까지 함께 불러오려면 필요합니다(우리가 만든 파일이라 안전).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 저장 형식이 바뀌었는데 옛 파일을 읽으면 KeyError 로 죽습니다. 미리 확인해 안내합니다.
    if ckpt.get("format_version") != 1:
        raise RuntimeError(
            f"체크포인트 형식 버전이 맞지 않습니다(기대 1, 실제 {ckpt.get('format_version')}).\n"
            "`python training.py` 로 다시 학습하세요."
        )
    src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
    cfg = ckpt["config"]
    model = GRUSeq2Seq(len(src_vocab), len(tgt_vocab),
                       emb_dim=cfg["emb_dim"], hidden_size=cfg["hidden_size"])
    # 하이퍼파라미터를 바꾸고 재학습을 잊으면 여기서 size mismatch 가 납니다.
    # 그 원인 불명 오류 대신 무엇을 해야 하는지 알려 줍니다.
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        raise RuntimeError(
            f"체크포인트 설정이 코드와 다릅니다(저장 {cfg}).\n"
            f"`python training.py` 로 다시 학습하세요.\n원본 오류: {e}"
        ) from e
    model.eval()                                  # 드롭아웃 끔 → 답이 매번 같음
    print("📂 불러오기 완료:", ckpt_path,
          f"(학습 최종 loss {ckpt['final_loss']:.4f} | 토큰 정확도 {ckpt['token_acc']:.3f})")
    return model, src_vocab, tgt_vocab


def cosine(a, b):
    """두 벡터가 얼마나 같은 방향인지. +1=완전 동일, 0=무관, -1=정반대.

    '길이'는 무시하고 '방향'만 봅니다. 요약 벡터가 원본에서 얼마나 딴 데로 갔는지 재기에
    딱 맞습니다 — 크기가 조금 커지거나 작아진 것은 중요하지 않으니까요.
    flatten() = 여러 줄로 된 표를 한 줄로 쭉 펴기(코사인은 1차원 벡터끼리 재는 것이므로).
    .item() = 숫자 1개짜리 텐서를 파이썬 float 으로 꺼내기.
    """
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def context_of(model, sv, question):
    """질문을 읽고 남은 '요약 벡터' 한 개만 꺼냅니다. (64,)

    인코더가 남긴 마지막 은닉 상태가 곧 요약 벡터입니다.
    """
    return encode_trace(model, sv, question)[0, -1]


def main():
    setup_korean_font()                           # 히트맵 축 한글 깨짐 방지
    model, src_vocab, tgt_vocab = load_model()
    print("✅ 체크포인트 로드 OK")

    # ── 5단계: 답을 한 단어씩 만들어 보기 ─────────────────────────────────
    q = MAIN_QUESTION
    # answer 가 5개를 한꺼번에 돌려줍니다. 왼쪽부터 순서대로 이름을 붙여 받는 것입니다.
    #   steps = [(지금까지 만든 답, 다음에 고른 단어), …] 생성 과정 기록
    #   ans   = 완성된 답 문자열
    #   hs    = 질문을 읽는 동안의 회의록 궤적  (1, 5, 64)
    #   rs/zs = 같은 구간의 리셋·갱신 게이트 값 (1, 5, 64)
    steps, ans, hs, rs, zs = answer(model, src_vocab, tgt_vocab, q)
    qs = tokenize(q)                              # 축 눈금·표에 쓸 질문 토큰 5개
    print(f"\n[5단계] 질문: {q}")
    print(f"  인코더가 질문 {len(qs)}단어를 차례로 읽어 요약 벡터 1개({hs.size(-1)}칸)를 만들었습니다.")
    print("  ⭐ 디코더는 지금부터 질문을 다시 볼 수 없습니다. 이 64칸이 전부입니다.")
    for seen, nxt in steps:
        print(f"  '{seen or '<sos>'}'  →  {nxt}")
    print("답:", ans)
    # 학습한 6개 답 중 하나가 나와야 합니다. 문자열을 직접 비교하는 대신 집합 포함으로 봅니다.
    assert ans in EXPECTED_ANSWERS, f"생성한 답 '{ans}' 이 학습한 6개 답 중 어디에도 없습니다."
    assert ans == "비가 올 것 같아", f"'먹구름이' 질문의 답이 '비가 올 것 같아' 가 아닙니다: '{ans}'"
    # <eos> 없이 max_len(20)까지 갔다면 학습이 덜 된 것입니다.
    assert steps[-1][1] == "<eos>", f"마지막 생성 토큰이 <eos> 가 아닙니다: '{steps[-1][1]}'"
    print("✅ 순차 생성 OK")

    # ── 6단계: 게이트가 한 일 들여다보기 ──────────────────────────────────
    # 어텐션 가중치가 없으므로 대신 게이트 값을 봅니다. 이것이 GRU 의 '속'입니다.
    print("\n[6단계] 게이트 기록 — 질문을 읽는 동안 회의록이 어떻게 바뀌었나")
    print("  토큰        리셋 r   갱신 z   요약 벡터 크기 |h|")
    z_means = []
    # enumerate = 번호와 값을 함께 꺼내기. t=0,1,2,… 가 몇 번째 단어인지입니다.
    for t, tok in enumerate(qs):
        # 64칸 게이트 값의 평균을 대표값으로 씁니다(칸별로 흩어진 값은 히트맵에서 봅니다).
        r_m, z_m = rs[0, t].mean().item(), zs[0, t].mean().item()
        z_means.append(z_m)
        # norm() = 벡터의 길이(피타고라스 정리를 64차원으로 확장한 것). 회의록에 적힌 양의 크기입니다.
        # f-string 서식: {tok:<10} 은 왼쪽 정렬 10칸, {r_m:.3f} 은 소수점 3자리로 맞춰 표를 정렬합니다.
        print(f"  {tok:<10} {r_m:.3f}    {z_m:.3f}        {hs[0, t].norm().item():.2f}")
    # z 가 낮을수록 회의록을 많이 고쳤다는 뜻입니다. 그런데 그냥 최저값을 찾으면 안 됩니다.
    # ⚠️ 첫 토큰(t=0)은 회의록이 백지(h=0)에서 시작하므로 **원래 z 가 낮게 나옵니다**.
    #    빈 종이에 처음 쓰는 것이니 '많이 고친다'가 당연한 것이고, 이건 그 단어가 중요해서가
    #    아닙니다. 실측에서도 '하늘에'(0.447)가 '먹구름이'(0.468)보다 낮게 나왔습니다.
    #    첫 칸을 빼고 비교해야 의미가 생깁니다 — 같은 함정이 |h_t - h_{t-1}| 에도 있습니다.
    # 읽는 법: 0번을 뺀 나머지(1~4번) 중 z 가 가장 작은 자리를 찾아 그 토큰 이름을 꺼냅니다.
    # min(..., key=...) = "이 기준으로 가장 작은 것"을 고르라는 뜻이고, 1+ 은 뺀 첫 칸을 되돌리는 보정입니다.
    lowest_z = qs[1 + min(range(len(z_means) - 1), key=lambda i: z_means[i + 1])]
    print(f"  ⭐ (첫 토큰 제외) z 가 가장 낮은 토큰: '{lowest_z}' — 기억을 가장 많이 고친 자리입니다.")
    print("     z 가 낮을수록 '옛 기억을 덜 지키고 새 정보를 많이 받았다'는 뜻입니다.")
    print("     ⚠️ 첫 토큰은 백지에서 시작해 원래 z 가 낮으므로 비교에서 뺐습니다.")
    print("     ℹ️ 게이트 값은 '관찰용'입니다. 검산은 아래 '기억 갈라짐' 숫자로 합니다 —")
    print("        게이트 최저 위치는 초기화에 따라 흔들리지만, 갈라짐은 구조에서 나옵니다.")

    if show_gates(q, rs, zs, save_path=HEATMAP_PATH):
        print("🖼️ 게이트 히트맵 저장:", HEATMAP_PATH)
        assert HEATMAP_PATH.exists() and HEATMAP_PATH.stat().st_size > 10_000, \
            f"히트맵이 비정상적으로 작습니다: {HEATMAP_PATH}"

    # 기억 갈라짐 검산 — 어텐션 없이 '주목한 곳'을 숫자로 확인하는 방법입니다.
    # 두 질문은 문장 틀이 같고 대상 단어만 다릅니다. 그래서 '하늘에'까지의 회의록은
    # 비트 단위로 똑같아야 하고, 대상 단어 자리에서 비로소 갈라져야 합니다.
    qa, qb = DIVERGE_PAIR
    hs_a, hs_b = encode_trace(model, src_vocab, qa), encode_trace(model, src_vocab, qb)
    assert torch.equal(hs_a[0, 0], hs_b[0, 0]), \
        "첫 토큰이 '하늘에' 로 같은데 은닉 상태가 다릅니다. 순차 처리가 깨졌습니다."
    # 두 궤적을 뺀 뒤 시점마다 길이를 재면 "그 시점에서 회의록이 얼마나 달라졌나"가 됩니다.
    # norm(dim=-1) = 마지막 축(64칸)만 하나의 길이로 접기 → (5, 64) 가 (5,) 가 됩니다.
    div = (hs_a[0] - hs_b[0]).norm(dim=-1)                # 시점별 회의록 차이 (5,)
    assert div[0].item() == 0.0, f"공통 토큰 자리에서 차이가 {div[0].item()} 입니다."
    assert div[1].item() > 1e-3, f"대상 단어 자리에서 기억이 갈라지지 않았습니다(Δ={div[1].item():.2e})."
    # 최대 지점의 '정확한 위치'는 초기화에 따라 흔들릴 수 있으므로 1 이상인지만 봅니다.
    assert div.argmax().item() >= 1, "기억 차이가 최대인 지점이 공통 토큰 자리입니다."
    assert (div[1:] > 1e-4).all().item(), "갈라진 차이가 이후 시점에서 사라졌습니다(기억 유지 실패)."
    diverge_at = div.argmax().item()
    print(f"✅ 기억 갈라짐 OK — {diverge_at}번 토큰 '{qs[diverge_at]}' 에서 최대 (Δ={div.max().item():.3f})")
    print("   시점별 차이:", " ".join(f"{d:.3f}" for d in div.tolist()))
    print("   📖 트랜스포머의 attention_heatmap.png 와 나란히 놓고 보세요.")
    print("      트랜스포머는 '답의 각 단어가 질문의 어디를 봤나'를 보여 주고,")
    print("      GRU 는 '질문을 읽는 동안 기억이 언제 크게 바뀌었나'만 보여 줍니다.")

    # ── 7단계: 키워드를 바꿔 답 비교하기 ──────────────────────────────────
    print("\n[7단계] 대상 단어만 바꾸면 답도 바뀝니다")
    got = {}                                      # {대상 단어: 생성한 답} 딕셔너리
    for cq, _ in DATA:
        # `_, a, *_` 읽는 법: 첫 값(steps)은 버리고, 두 번째(답)만 a 로 받고,
        # 나머지 3개(hs/rs/zs)는 `*_` 로 한꺼번에 버립니다. 여기서는 답만 필요하니까요.
        _, a, *_ = answer(model, src_vocab, tgt_vocab, cq)
        # tokenize(cq)[1] = 질문의 두 번째 토큰(=바뀌는 '대상' 단어). :<8 은 왼쪽 정렬 8칸 폭.
        print(f"  {tokenize(cq)[1]:<8} → {a}")
        got[tokenize(cq)[1]] = a
    # 6개 답이 모두 달라야 합니다. 겹치면 모델이 대상 단어를 구분하지 못한 것입니다.
    assert len(set(got.values())) == 6, f"대상 단어를 바꿨는데 답이 겹칩니다: {got}"
    assert got["먹구름이"] == "비가 올 것 같아" and got["별이"] == "밤이 깊었나 봐"
    print("✅ 대조 검증 OK — 6쌍 모두 정답")

    # ── 8단계: 기억 씻김 — 이 실습의 결론 ─────────────────────────────────
    # 질문 '뒤'에 뜻 없는 단어를 덧붙여 핵심어를 밀어냅니다. 새 단어를 쓰지 않으므로
    # <unk> 문제가 아니고, 배치가 1문장이라 패딩 문제도 아닙니다. 순수하게 '길이' 효과입니다.
    print(f"\n[8단계] ⚠️ 기억 씻김 — 질문 뒤에 '{FILLER_WORD}' 를 붙이면?")
    base_ctx = context_of(model, src_vocab, q)    # 군더더기 없는 원본의 요약 벡터 (비교 기준)
    sims = []
    for k in FILLER_COUNTS:
        # 문자열 * 숫자 = 그만큼 반복. (" 하늘에" * 3) → " 하늘에 하늘에 하늘에"
        long_q = q + ((" " + FILLER_WORD) * k)
        _, a, *_ = answer(model, src_vocab, tgt_vocab, long_q)
        # 늘어난 질문의 요약 벡터가 원본과 얼마나 같은 방향인지 잽니다.
        sim = cosine(context_of(model, src_vocab, long_q), base_ctx)
        sims.append(sim)
        # 삼항 표현식: 조건이 참이면 앞의 값, 거짓이면 뒤의 값을 씁니다.
        mark = "✅" if a == "비가 올 것 같아" else "❌"
        print(f"  '{FILLER_WORD}' {k:>2}번 (길이 {len(tokenize(long_q)):>2}) → {a:<12} {mark}"
              f"   요약 벡터 유사도 {sim:+.3f}")
    print("   ⭐ 회의록은 한 장뿐입니다. 뒷내용을 덧쓰는 동안 '먹구름이'가 씻겨 나갔습니다.")
    print("      트랜스포머라면 질문 어디에 있든 '먹구름이'를 직접 다시 볼 수 있었습니다.")
    # 답이 무너지는지는 시드에 따라 흔들릴 수 있지만, 요약 벡터가 원본에서 멀어지는 것은
    # 구조에서 나오는 현상이라 항상 재현됩니다. 그래서 이 숫자를 주 증거로 씁니다.
    assert sims[0] > 0.999, f"군더더기 0개인데 유사도가 {sims[0]:.3f} 입니다(같은 질문이어야 함)."
    assert sims[-1] < sims[0], "질문을 늘렸는데 요약 벡터가 원본에서 멀어지지 않았습니다."
    print(f"✅ 병목 확인 OK — 유사도가 {sims[0]:+.3f} 에서 {sims[-1]:+.3f} 로 무너졌습니다")
    # 길이 제한이 없다는 것도 함께 보여 줍니다(트랜스포머는 max_len 을 넘으면 에러가 납니다).
    print("   ℹ️ GRU 는 문장이 길어져도 에러가 나지 않습니다 — 위치인코딩 표가 없으니까요.")

    # ── 9단계: 최근성 편향 ────────────────────────────────────────────────
    print("\n[9단계] ⚠️ 최근성 편향 — 키워드를 두 개 넣으면 어느 쪽이 이길까요?")
    order_answers = []
    for oq in ORDER_QUESTIONS:
        _, a, *_ = answer(model, src_vocab, tgt_vocab, oq)
        # toks[1] 은 두 번째 단어(앞 키워드), toks[2] 는 세 번째 단어(뒤 키워드)입니다.
        # 두 질문은 이 둘의 순서만 서로 바꾼 것이고, 나머지는 완전히 같습니다.
        toks = tokenize(oq)
        print(f"  {oq} → {a}")
        print(f"     (앞 키워드 '{toks[1]}' / 뒤 키워드 '{toks[2]}')")
        order_answers.append(a)
    print("   ⭐ GRU 는 순서대로 덧쓰기 때문에 '나중에 들은 말'에 유리합니다.")
    print("      어텐션은 순서가 아니라 관련도로 고르므로 이런 편향이 훨씬 약합니다.")
    # 순서만 바꿨는데 답이 달라진다는 것 자체가 '순서에 끌려다닌다'는 증거입니다.
    assert order_answers[0] != order_answers[1], \
        f"키워드 순서를 바꿨는데 답이 같습니다: {order_answers}"
    print("✅ 최근성 편향 확인 OK — 순서만 바꿨는데 답이 달라졌습니다")

    # ── 마무리: 결정성 확인 + 30초 요약 + 회귀 지문 ────────────────────────
    # eval() 이 제대로 걸렸으면 같은 질문에 항상 같은 답이 나와야 합니다.
    # 3번 물어본 답을 집합에 담습니다. 집합은 중복을 자동으로 없애므로,
    # 답이 항상 같으면 원소가 1개이고 하나라도 다르면 2개 이상이 됩니다.
    # ⚠️ 드롭아웃이 켜져 있으면(eval() 누락) 여기서 갈립니다 — 그래서 이 검사를 넣었습니다.
    reps = {answer(model, src_vocab, tgt_vocab, q)[1] for _ in range(3)}
    assert len(reps) == 1, f"같은 질문에 답이 3회 중 갈렸습니다: {reps}. model.eval() 을 확인하세요."
    print("\n✅ 결정성 OK — 같은 질문 3회 모두 같은 답")
    print("✅ 실습 완료! 30초 요약")
    print("   1) GRU 는 은닉 상태 '한 장'을 게이트 두 개로 고쳐 씁니다.")
    print("   2) 디코더는 질문 원문이 아니라 요약 벡터 한 개만 받습니다(정보 병목).")
    print("   3) 그 병목을 없애려고 나온 것이 어텐션이고, 순차 루프를 없애려고 나온 것이 트랜스포머입니다.")
    # 회귀 지문: 코드를 고친 뒤 이 한 줄이 그대로인지만 보면 됩니다. 전부 이산값이라
    # 플랫폼·torch 버전이 달라도 흔들리지 않습니다.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🔖 회귀 지문: params={n_params} | vocab={len(src_vocab)}/{len(tgt_vocab)} "
          f"| steps={len(steps)} | ans={ans} | 대조={len(set(got.values()))}종 | 갈라짐={diverge_at}")


if __name__ == "__main__":
    main()
