# -*- coding: utf-8 -*-
"""모델 훈련 — `example-explain.ipynb` 의 1·2·4단계를 스크립트로 옮긴 것.

📖 상세 설명: https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/hands-on/mini-transformer/training.html

[한 줄 요약] 눈 감고 다트 던지기 연습입니다. 던지고, 얼마나 빗나갔는지 듣고, 팔 각도를 조금 고칩니다.

하는 일
    1단계  DATA 6쌍 확인
    2단계  질문/답 단어장(src_vocab / tgt_vocab) 만들기
    4단계  배치 만들기 → 모델 생성 → 400에폭 학습 → 체크포인트 저장

학습 결과(모델 가중치 + 단어장 + max_len)를 `mini_transformer.pt` 로 저장합니다.
추론은 `reasoning.py` 가 이 파일을 읽어서 수행합니다.

실행:  python training.py
"""
from pathlib import Path

import torch
from torch import nn

from mini_transformer import (
    DATA,
    Transformer,
    Vocab,
    build_batches,
    make_decoder_mask,
    make_padding_mask,
)

# 학습 결과를 저장할 경로. reasoning.py 가 같은 경로에서 읽습니다.
CKPT_PATH = Path(__file__).with_name("mini_transformer.pt")


def main():
    # ── 1단계: 데이터 확인 ────────────────────────────────────────────────
    for q, a in DATA[:3]:
        print(q, "→", a)
    assert len(DATA) == 6
    print("✅ 데이터 OK")

    # ── 2단계: 단어장 만들기 ──────────────────────────────────────────────
    # 질문과 답은 쓰는 단어가 하나도 겹치지 않으므로 단어장을 각각 따로 만듭니다.
    src_vocab = Vocab([q for q, _ in DATA])         # 질문 단어장
    tgt_vocab = Vocab([a for _, a in DATA])         # 답 단어장
    print("질문 단어", len(src_vocab), "| 답 단어", len(tgt_vocab))
    assert src_vocab.pad_id == 0 and tgt_vocab.sos_id == 1
    print("✅ 사전 OK")

    # ── 4단계: 모델 만들고 학습하기 ───────────────────────────────────────
    # 같은 결과가 나오도록 난수 시드를 고정합니다(가중치 초기화·드롭아웃이 매번 동일해짐).
    torch.manual_seed(42)                                    # 재현성(같은 결과)
    # 질문/답 배치 텐서 준비. 실측 shape: (6, 5)와 (6, 6).
    src_ids, tgt_ids = build_batches(DATA, src_vocab, tgt_vocab)
    # 위치인코딩 표가 넉넉하도록 데이터 최대 길이 + 여유로 잡습니다. 실측: max(5, 6, 20)+2 = 22.
    max_len = max(src_ids.size(1), tgt_ids.size(1), 20) + 2
    # 단어장 크기를 넘겨 모델 생성. 이 한 줄로 만들어지는 학습 대상 가중치가 총 235,520개입니다.
    model = Transformer(len(src_vocab), len(tgt_vocab), max_len=max_len)
    print("학습 대상 파라미터", sum(p.numel() for p in model.parameters()), "개")
    # Adam: 가중치를 자동으로 갱신해 주는 옵티마이저. 3e-4 는 '아주 조금씩' 고치라는 뜻입니다.
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    # ignore_index 로 <pad> 자리는 손실 계산에서 제외합니다(발판은 '사람 수'에 안 들어감).
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    # 티처 포싱: 디코더 입력은 <sos> 부터, 정답은 한 칸 뒤. 실측 0행: [1,13,15,5,4] / [13,15,5,4,2]
    dec_in, dec_tgt = tgt_ids[:, :-1], tgt_ids[:, 1:]        # 입력은 <sos>부터, 정답은 한 칸 뒤
    # 질문 쪽 패딩 마스크는 400번 내내 바뀌지 않으므로 루프 밖에서 한 번만 계산합니다.
    src_mask = make_padding_mask(src_ids, src_vocab.pad_id)
    model.train()                                            # 학습 모드(드롭아웃 켜짐)

    # 6쌍을 한 번 보여주는 걸로는 못 외웁니다. 같은 6쌍을 400번 다시 보여주며 조금씩 다듬는 것입니다.
    for ep in range(1, 401):
        # 디코더 마스크는 미래 가림 + 패딩 가림을 함께. dec_in 길이에 맞춰 매번 생성.
        tgt_mask = make_decoder_mask(dec_in, tgt_vocab.pad_id)
        logits = model(src_ids, dec_in, src_mask, tgt_mask)                    # (1) 예측
        # (B,T,V) → (B*T,V), (B,T) → (B*T,) 로 펼쳐 CrossEntropy 계산. <pad> 칸은 빠집니다.
        loss = crit(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))  # (2) 정답과 비교
        # backward = 어느 방향으로 틀렸는지 기울기 계산, step = 그 반대 방향으로 한 걸음 이동.
        # zero_grad 를 빼면 지난 기울기가 계속 더해져 엉뚱한 방향으로 갑니다.
        opt.zero_grad(); loss.backward(); opt.step()                          # (3) 가중치 수정
        if ep % 100 == 0 or ep == 1:
            print(f"epoch {ep:4d} | loss {loss.item():.4f}")

    # 체크포인트: 학습이 충분히 됐는지 최종 loss 가 0.5 미만인지 확인.
    assert loss.item() < 0.5
    print("✅ 학습 완료! 최종 loss", round(loss.item(), 4))

    # ── 저장 ──────────────────────────────────────────────────────────────
    # 단어장까지 함께 저장해야 추론에서 같은 번호 체계를 쓸 수 있습니다.
    # max_len 도 저장합니다 — 모델 생성 인자가 달라지면 state_dict 모양이 안 맞습니다.
    torch.save({
        "model_state": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "max_len": max_len,
        "final_loss": loss.item(),
    }, CKPT_PATH)
    print("💾 저장 완료:", CKPT_PATH)


if __name__ == "__main__":
    main()
