# -*- coding: utf-8 -*-
"""추론 수행 — `example-explain.ipynb` 의 5·6·7단계를 스크립트로 옮긴 것.

📖 상세 설명: https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/hands-on/mini-transformer/reasoning.html

하는 일
    5단계  질문 하나를 넣어 답을 한 단어씩 만드는 과정을 출력
    6단계  크로스어텐션 히트맵을 PNG 로 저장 ('먹구름이' 열이 가장 밝으면 성공 🌧️)
    7단계  '대상' 단어만 바꿔 답이 달라지는지 대조

`training.py` 가 만든 `mini_transformer.pt` 가 먼저 있어야 합니다.

실행:  python training.py  &&  python reasoning.py
"""
from pathlib import Path

import torch

from mini_transformer import (
    Transformer,
    answer,
    setup_korean_font,
    show_attention,
    tokenize,
)

# training.py 가 저장한 체크포인트 / 히트맵을 내보낼 경로.
CKPT_PATH = Path(__file__).with_name("mini_transformer.pt")
HEATMAP_PATH = Path(__file__).with_name("attention_heatmap.png")

# 7단계 대조용 질문 — 문장 틀은 그대로 두고 '대상' 단어만 바꿉니다.
CONTRAST_QUESTIONS = [
    "하늘에 먹구름이 보이면 뭐가 생각나",
    "하늘에 별이 보이면 뭐가 생각나",
    "하늘에 해가 보이면 뭐가 생각나",
]


def load_model(ckpt_path=CKPT_PATH):
    """체크포인트에서 모델과 단어장을 되살립니다.

    단어장을 함께 저장해 둔 이유: 번호 체계가 학습 때와 한 칸이라도 어긋나면
    같은 가중치라도 완전히 다른 답이 나옵니다.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {ckpt_path}\n먼저 `python training.py` 를 실행하세요."
        )
    # weights_only=False: Vocab 객체까지 함께 불러오려면 필요합니다(우리가 만든 파일이라 안전).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
    # max_len 을 저장값 그대로 넘겨야 위치인코딩 버퍼 모양이 맞습니다.
    model = Transformer(len(src_vocab), len(tgt_vocab), max_len=ckpt["max_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()                                  # 드롭아웃 끔 → 답이 매번 같음
    print("📂 불러오기 완료:", ckpt_path, f"(학습 최종 loss {ckpt['final_loss']:.4f})")
    return model, src_vocab, tgt_vocab


def main():
    setup_korean_font()                           # 히트맵 축 한글 깨짐 방지
    model, src_vocab, tgt_vocab = load_model()

    # ── 5단계: 답을 한 단어씩 만들어 보기 ─────────────────────────────────
    q = "하늘에 먹구름이 보이면 뭐가 생각나"
    # answer 가 (한 단어씩 만든 과정, 최종 답, 크로스어텐션)을 돌려줍니다.
    steps, ans, cross = answer(model, src_vocab, tgt_vocab, q)
    print("\n질문:", q)
    for seen, nxt in steps:
        print(f"  '{seen or '<sos>'}'  →  {nxt}")
    print("답:", ans)

    # ── 6단계: 어텐션 히트맵 ──────────────────────────────────────────────
    # '먹구름이' 열이 가장 환하면 모델이 그 단어에 주목해 '비'를 떠올린 것!
    show_attention(q, ans, cross, save_path=HEATMAP_PATH)
    print("\n🖼️ 히트맵 저장:", HEATMAP_PATH)
    # 어느 질문 토큰을 가장 크게 봤는지 숫자로도 확인합니다(히트맵 검산용).
    qs = tokenize(q)
    top = cross[: len(qs), : len(qs)].mean(0).argmax().item()
    print(f"   가장 크게 주목한 질문 토큰: '{qs[top]}'")

    # ── 7단계: 키워드를 바꿔 답 비교하기 ──────────────────────────────────
    print("\n[대조] 대상 단어만 바꾸면 답도 바뀝니다")
    for q in CONTRAST_QUESTIONS:
        # 답만 필요하므로 steps·cross 는 _ 로 버립니다.
        _, a, _ = answer(model, src_vocab, tgt_vocab, q)
        # tokenize(q)[1] = 질문의 두 번째 토큰(=바뀌는 '대상' 단어). :<6 은 왼쪽 정렬 6칸 폭.
        print(f"  {tokenize(q)[1]:<6} → {a}")


if __name__ == "__main__":
    main()
