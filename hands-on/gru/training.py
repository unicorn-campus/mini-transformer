# -*- coding: utf-8 -*-
"""GRU 모델 훈련 — 미니 트랜스포머 `training.py` 와 같은 데이터·같은 절차를 GRU 로 수행합니다.

🖼️ 그림(트랜스포머 대비): https://raw.githubusercontent.com/unicorn-campus/mini-transformer/main/hands-on/gru/images/gru-08-vs-transformer.webp
📖 학습 루프 설명: https://htmlpreview.github.io/?https://github.com/unicorn-campus/mini-transformer/blob/main/explain/training-loop.html

[한 줄 요약] 눈 감고 다트 던지기입니다. 다만 트랜스포머는 6발을 한 번에 던졌고,
GRU 는 앞의 다트가 꽂힌 뒤에야 다음 다트를 던질 수 있습니다.

하는 일
    1단계  DATA 6쌍 확인 (미니 트랜스포머와 같은 데이터라는 것을 눈으로 확인)
    2단계  질문/답 단어장(src_vocab / tgt_vocab) 만들기
    4단계  배치 만들기 → 모델 생성 → 400에폭 학습 → 체크포인트 저장

⭐ 미니 트랜스포머의 training.py 와 나란히 놓고 비교해 보세요. 데이터·단어장·티처 포싱·
   손실 함수·옵티마이저·에폭 수·학습률·시드까지 전부 같고, 바뀐 것은
   `Transformer(...)` 가 `GRUSeq2Seq(...)` 로 바뀐 **한 줄**입니다.
   그 한 줄 안에서 파라미터가 235,520개에서 64,256개로 줄고, 대신 for 루프가 생깁니다.
   나머지를 전부 고정했기 때문에, 두 실습의 결과 차이는 100% '구조 때문'입니다.

학습 결과(모델 가중치 + 단어장 + 설정)를 `mini_gru.pt` 로 저장합니다.
추론은 `reasoning.py` 가 이 파일을 읽어서 수행합니다.

실행:  python training.py
"""
import math
import time
from pathlib import Path

import torch
from torch import nn

from mini_gru import (
    DATA,
    GRUSeq2Seq,
    Vocab,
    build_batches,
)

# 학습 결과를 저장할 경로. reasoning.py 가 같은 경로에서 읽습니다.
# Path(__file__) 을 쓰면 어느 폴더에서 실행해도 항상 이 파일 옆에 저장됩니다.
CKPT_PATH = Path(__file__).with_name("mini_gru.pt")

# 하이퍼파라미터를 한곳에 모아 둡니다. 값은 전부 미니 트랜스포머와 동일합니다.
EMB_DIM, HIDDEN_SIZE = 64, 64      # 뜻벡터 칸수 / 회의록 칸수 (트랜스포머 d_model=64)
EPOCHS, LR = 400, 3e-4             # 같은 6쌍을 400번 다시 보여 주며 조금씩 다듬음
# 수렴 판정 기준 3종. 아래 [검증] 절 주석에 각 값의 근거가 있습니다.
TRAIN_LOSS_GATE = 0.35             # 드롭아웃이 켜진 '학습 모드' loss 기준
EVAL_LOSS_GATE = 0.10              # 드롭아웃을 끈 '진짜 실력' 기준


def main():
    # ── 1단계: 데이터 확인 ────────────────────────────────────────────────
    for q, a in DATA[:3]:
        print(q, "→", a)
    assert len(DATA) == 6, f"DATA 는 6쌍이어야 합니다(현재 {len(DATA)}쌍)."
    # 답이 겹치면 7단계 대조 검증이 무의미해집니다(어느 질문이든 같은 답이 나올 수 있으므로).
    assert len({a for _, a in DATA}) == 6, "답에 중복이 있습니다."
    print("✅ 데이터 OK — 미니 트랜스포머와 똑같은 6쌍입니다")

    # ── 2단계: 단어장 만들기 ──────────────────────────────────────────────
    # 질문과 답은 쓰는 단어가 하나도 겹치지 않으므로 단어장을 각각 따로 만듭니다.
    src_vocab = Vocab([q for q, _ in DATA])         # 질문 단어장
    tgt_vocab = Vocab([a for _, a in DATA])         # 답 단어장
    print("질문 단어", len(src_vocab), "| 답 단어", len(tgt_vocab))
    assert src_vocab.pad_id == 0 and tgt_vocab.sos_id == 1
    assert (len(src_vocab), len(tgt_vocab)) == (14, 18), \
        f"단어장 크기가 (14, 18) 이 아닙니다: ({len(src_vocab)}, {len(tgt_vocab)})"
    print("✅ 사전 OK")

    # ── 4단계: 모델 만들고 학습하기 ───────────────────────────────────────
    # 같은 결과가 나오도록 난수 시드를 고정합니다(가중치 초기화·드롭아웃이 매번 동일해짐).
    torch.manual_seed(42)                                    # 재현성(같은 결과)
    # 질문/답 배치 텐서 준비. 실측 shape: (6, 5)와 (6, 6).
    src_ids, tgt_ids = build_batches(DATA, src_vocab, tgt_vocab)
    # ⚠️ GRU 인코더에는 패딩 마스크가 없어 <pad> 도 그대로 한 스텝을 차지합니다.
    #    CrossEntropyLoss(ignore_index=pad_id) 는 '점수 계산'에서만 <pad> 를 빼 줄 뿐,
    #    은닉 상태는 <pad> 를 읽는 동안에도 계속 바뀝니다 — 손실에서 뺐다고 기억이 깨끗한 게
    #    아닙니다. 지금 질문 6개는 모두 5단어라 패딩이 0칸이어서 문제가 없습니다.
    #    길이가 다른 질문을 추가하면 pack_padded_sequence 가 필요합니다(이 실습 범위 밖).
    #    조용히 성능이 망가지는 것을 막기 위해 여기서 즉시 에러로 잡습니다.
    assert (src_ids != src_vocab.pad_id).all(), "질문 길이가 서로 달라 <pad> 가 끼었습니다."
    print("✅ 패딩 없음 OK                        질문 6개가 모두 5단어입니다")

    # 티처 포싱: 디코더 입력은 <sos> 부터, 정답은 한 칸 뒤. 실측 0행: [1,13,15,5,4] / [13,15,5,4,2]
    dec_in, dec_tgt = tgt_ids[:, :-1], tgt_ids[:, 1:]        # 입력은 <sos>부터, 정답은 한 칸 뒤
    assert torch.equal(dec_in[:, 1:], dec_tgt[:, :-1]), \
        "티처 포싱 오프셋이 어긋났습니다. dec_in=tgt[:, :-1], dec_tgt=tgt[:, 1:] 인지 확인하세요."
    print(f"질문 배치 {tuple(src_ids.shape)} | 답 배치 {tuple(tgt_ids.shape)} "
          f"| 디코더 입력 {tuple(dec_in.shape)}")

    # 단어장 크기를 넘겨 모델 생성. 이 한 줄이 트랜스포머와 다른 유일한 곳입니다.
    model = GRUSeq2Seq(len(src_vocab), len(tgt_vocab),
                       emb_dim=EMB_DIM, hidden_size=HIDDEN_SIZE)
    # model.parameters() = 학습으로 바뀔 수 있는 모든 숫자 덩어리를 하나씩 꺼내 줍니다.
    # p.numel() = 그 덩어리 안의 숫자 개수(number of elements). 전부 더하면 총 파라미터 수입니다.
    # ⭐ 가중치 공유로 같은 표를 두 곳에서 쓰지만, parameters() 는 같은 것을 두 번 세지 않습니다.
    n_params = sum(p.numel() for p in model.parameters())
    print("학습 대상 파라미터", n_params, "개")
    assert n_params == 64256, f"파라미터 수가 64,256 개가 아닙니다({n_params}). 하이퍼파라미터를 확인하세요."
    print("   ⭐ 미니 트랜스포머는 235520 개였습니다. GRU 는 그 27% 크기입니다.")
    print(f"   ⭐ 대신 문장 하나를 읽으려면 for 루프를 {src_ids.size(1)}번 돌아야 합니다"
          f"(트랜스포머는 한 번에 읽었습니다).")
    assert model.out.weight is model.decoder.emb.weight, "가중치 공유가 끊어졌습니다."
    print("✅ 모델 조립 OK")

    # Adam: 가중치를 자동으로 갱신해 주는 옵티마이저. 3e-4 는 '아주 조금씩' 고치라는 뜻입니다.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # ignore_index 로 <pad> 자리는 손실 계산에서 제외합니다(발판은 '사람 수'에 안 들어감).
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)
    model.train()                                            # 학습 모드(드롭아웃 켜짐)

    # 6쌍을 한 번 보여주는 걸로는 못 외웁니다. 같은 6쌍을 400번 다시 보여주며 조금씩 다듬는 것입니다.
    # perf_counter = 초 단위 스톱워치. 시계 시각이 아니라 '흐른 시간'을 재는 용도입니다.
    # ⭐ 이 시간을 재는 이유가 있습니다. GRU 는 파라미터가 트랜스포머의 27%인데도 순차 루프
    #    때문에 시간이 비례해서 줄지 않습니다. "작다 ≠ 빠르다" 를 숫자로 보여 주려는 것입니다.
    started = time.perf_counter()
    loss_hist = []          # 에폭별 loss 를 모아 둡니다(아래 상대 감소율 검사에 씁니다)
    for ep in range(1, EPOCHS + 1):
        # ⭐ 마스크가 없습니다. 트랜스포머는 매 에폭 make_decoder_mask 를 만들어야 했지만,
        #    GRU 는 미래를 아직 계산조차 안 했으므로 가릴 것이 없습니다.
        logits = model(src_ids, dec_in)                                        # (1) 예측
        # (B,T,V) → (B*T,V), (B,T) → (B*T,) 로 펼쳐 CrossEntropy 계산. <pad> 칸은 빠집니다.
        loss = crit(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))  # (2) 정답과 비교
        # backward = 어느 방향으로 틀렸는지 기울기 계산, step = 그 반대 방향으로 한 걸음 이동.
        # ⚠️ zero_grad 를 빼면 지난 기울기가 계속 더해집니다. GRU 는 for 루프로 같은 셀을
        #    여러 번 지나가므로 기울기가 쌓이는 속도가 트랜스포머보다 빠릅니다.
        opt.zero_grad(); loss.backward(); opt.step()                          # (3) 가중치 수정
        # 학습이 폭발(nan/inf)하면 이후 로그가 전부 무의미해지므로 즉시 잡습니다.
        assert torch.isfinite(loss), \
            f"epoch {ep} 에서 loss 가 {loss.item()} 입니다. lr 을 3e-4 로 낮추세요."
        loss_hist.append(loss.item())
        if ep == 1:
            assert logits.shape == (6, 5, 18), f"logits shape 이상: {tuple(logits.shape)}"
            # 아무것도 모르는 상태에서 답 단어 18개를 '똑같은 확률로' 추측하면
            # loss = ln(18) = 2.890 이 됩니다. 그런데 실측 시작값은 약 4.97 로 이보다 높습니다.
            # ⭐ 왜 이론값보다 높은가 — 가중치 공유 때문입니다. 출력층이 임베딩 표를 그대로
            #    쓰는데, 그 표는 N(0,1) 로 초기화되어 값이 큽니다. 그래서 시작 점수가 균등하지
            #    않고 들쭉날쭉해 loss 가 더 높게 출발합니다(미니 트랜스포머도 같은 이유로 높습니다).
            #    '틀린 것'이 아니라 '이론값은 가중치 공유가 없을 때의 값'이라고 이해하면 됩니다.
            # 그래서 여기서는 정확한 값을 재는 대신 **폭발하지 않았는지**만 넉넉히 확인합니다.
            # 정답 정렬은 위쪽 `dec_in[:, 1:] == dec_tgt[:, :-1]` assert 가 정확히 잡아 줍니다.
            assert 1.0 < loss_hist[0] < 8.0, (
                f"1에폭 loss 가 {loss_hist[0]:.4f} 입니다. 균등 추측값 ln(18)="
                f"{math.log(18):.3f} 에서 너무 멀리 떨어져 있습니다. "
                "티처 포싱 오프셋·ignore_index·reshape 축을 확인하세요."
            )
        if ep % 100 == 0 or ep == 1:
            print(f"epoch {ep:4d} | loss {loss.item():.4f}")
    elapsed = time.perf_counter() - started

    # ── 검증: 수렴 게이트 3단 ─────────────────────────────────────────────
    # 트랜스포머 실습은 `loss < 0.5` 한 줄이었습니다. 그런데 드롭아웃이 켜진 학습 모드 loss 에는
    # 바닥(floor)이 생겨 기준을 느슨하게 잡아야 하므로 '덜 배운 상태'를 놓칠 수 있습니다.
    # 그래서 여기서는 세 각도로 봅니다.
    train_loss = loss_hist[-1]
    assert train_loss < TRAIN_LOSS_GATE, (                    # ① 관측: 드롭아웃 켜진 값
        f"학습 loss 가 {train_loss:.4f} 로 기준 {TRAIN_LOSS_GATE} 이상입니다.\n"
        "  진단 순서(한 번에 하나만): ① EPOCHS 800 → ② LR 1e-3 → ③ HIDDEN_SIZE 128"
    )
    # ② 상대 감소율. '절대 몇 점'이 아니라 '시작값의 몇 %까지 내려갔나'를 봅니다.
    #    컴퓨터마다 소수점 아래가 미세하게 달라도 이 비율은 흔들리지 않아 더 믿을 수 있습니다.
    assert train_loss < loss_hist[0] * 0.15, (
        f"loss 가 시작값의 15% 아래로 못 내려갔습니다({loss_hist[0]:.4f} → {train_loss:.4f})."
    )
    model.eval()                                             # 드롭아웃 끈 '진짜 실력'
    # no_grad = "이건 채점만 할 거니 기울기 계산은 하지 마" 라는 뜻입니다. 메모리와 시간을 아낍니다.
    with torch.no_grad():
        ev = model(src_ids, dec_in)
        eval_loss = crit(ev.reshape(-1, ev.size(-1)), dec_tgt.reshape(-1)).item()
        # 토큰 정확도: 1등으로 고른 단어가 정답과 같은 비율. <pad> 칸은 셈에서 뺍니다.
        # keep = 정답이 <pad> 가 아닌 자리만 True 인 표(마스크). (6, 5) 모양입니다.
        keep = dec_tgt != tgt_vocab.pad_id
        # argmax(-1) = 마지막 축(단어 18개)에서 점수 1등 번호 고르기 → (6, 5)
        # [keep] = True 인 자리만 골라 1차원으로 쭉 펴기. 그 뒤 정답과 비교해 평균을 냅니다.
        # ⭐ 이 값이 1.000 이어야 '6쌍을 정말 외웠다'가 됩니다. loss 숫자보다 이쪽이 확실합니다.
        acc = (ev.argmax(-1)[keep] == dec_tgt[keep]).float().mean().item()
    assert eval_loss < EVAL_LOSS_GATE, \
        f"평가 loss {eval_loss:.4f} 가 기준 {EVAL_LOSS_GATE} 이상입니다."
    # ③ 행위 게이트: argmax 는 이산값이라 플랫폼·버전이 달라도 흔들리지 않습니다.
    #    숫자 기준보다 이쪽이 '정말 외웠는가'를 더 정확히 판정합니다.
    assert acc == 1.0, \
        f"티처 포싱 토큰 정확도가 {acc:.3f} 입니다(1.000 이어야 함). 6쌍 암기가 덜 됐습니다."
    print(f"✅ 학습 완료! 학습 loss {train_loss:.4f} | 평가 loss {eval_loss:.4f} "
          f"| 토큰 정확도 {acc:.3f}   (약 {elapsed:.1f}초)")
    print(f"   for 루프가 돈 횟수: {EPOCHS}에폭 × (질문 {src_ids.size(1)}칸 + 답 "
          f"{dec_in.size(1)}칸) = {EPOCHS * (src_ids.size(1) + dec_in.size(1))}번")

    # ── 저장 ──────────────────────────────────────────────────────────────
    # 단어장까지 함께 저장해야 추론에서 같은 번호 체계를 쓸 수 있습니다.
    # config 도 저장합니다 — 나중에 EMB_DIM 이나 HIDDEN_SIZE 를 바꾸고 재학습을 잊으면
    # reasoning.py 가 "설정이 다릅니다" 라고 친절히 알려 줄 수 있습니다.
    # ⭐ 트랜스포머는 여기에 max_len 도 저장했습니다. GRU 에는 그 키가 없습니다 —
    #    위치인코딩 표가 없어서 문장 길이에 따라 모양이 바뀌는 파라미터가 하나도 없기 때문입니다.
    #    (그래서 GRU 는 학습 때보다 긴 문장을 넣어도 에러가 나지 않습니다)
    torch.save({
        "format_version": 1,
        "config": {"emb_dim": EMB_DIM, "hidden_size": HIDDEN_SIZE},
        "model_state": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "final_loss": train_loss,
        "eval_loss": eval_loss,
        "token_acc": acc,
    }, CKPT_PATH)
    # 파일이 정말 만들어졌는지, 내용이 비어 있지 않은지 확인합니다(Ctrl+C 로 끊긴 경우 대비).
    assert CKPT_PATH.exists() and CKPT_PATH.stat().st_size > 100_000, \
        "체크포인트가 비정상적으로 작습니다."
    print("💾 저장 완료:", CKPT_PATH, f"({CKPT_PATH.stat().st_size // 1024} KB)")
    print("   다음 단계:  python reasoning.py")


if __name__ == "__main__":
    main()
