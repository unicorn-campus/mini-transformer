"use client";

import { useEffect, useMemo, useState } from "react";

type WeatherSample = {
  key: string;
  label: string;
  subject: string;
  answer: string;
  sourceIds: number[];
  answerIds: number[];
  answerTokens: string[];
  accent: string;
};

const samples: WeatherSample[] = [
  {
    key: "cloud",
    label: "먹구름",
    subject: "먹구름이",
    answer: "비가 올 것 같아",
    sourceIds: [12, 6, 10, 8, 11],
    answerIds: [13, 15, 5, 4],
    answerTokens: ["비가", "올", "것", "같아"],
    accent: "#3d77ea",
  },
  {
    key: "star",
    label: "별",
    subject: "별이",
    answer: "밤이 깊었나 봐",
    sourceIds: [12, 9, 10, 8, 11],
    answerIds: [11, 8, 12],
    answerTokens: ["밤이", "깊었나", "봐"],
    accent: "#7c62c9",
  },
  {
    key: "sun",
    label: "해",
    subject: "해가",
    answer: "아침이 밝았구나",
    sourceIds: [12, 13, 10, 8, 11],
    answerIds: [14, 10],
    answerTokens: ["아침이", "밝았구나"],
    accent: "#e4922f",
  },
  {
    key: "rainbow",
    label: "무지개",
    subject: "무지개가",
    answer: "비가 그쳤나 봐",
    sourceIds: [12, 7, 10, 8, 11],
    answerIds: [13, 7, 12],
    answerTokens: ["비가", "그쳤나", "봐"],
    accent: "#d85a7f",
  },
  {
    key: "snow",
    label: "눈송이",
    subject: "눈송이가",
    answer: "겨울이 왔구나",
    sourceIds: [12, 5, 10, 8, 11],
    answerIds: [6, 16],
    answerTokens: ["겨울이", "왔구나"],
    accent: "#399fb0",
  },
  {
    key: "sunset",
    label: "노을",
    subject: "노을이",
    answer: "저녁이 되었네",
    sourceIds: [12, 4, 10, 8, 11],
    answerIds: [17, 9],
    answerTokens: ["저녁이", "되었네"],
    accent: "#c8683c",
  },
];

const processGroups = [
  {
    id: "develop",
    label: "DEVELOP · 모델 만들기",
    description: "데이터와 구조를 코드로 개발",
    items: [
      ["data", "01", "학습 데이터 설계"],
      ["tokens", "02", "단어장 · 배치 준비"],
      ["encode", "03", "인코더 구현"],
      ["decode", "04", "디코더 구현"],
    ],
  },
  {
    id: "offline",
    label: "OFFLINE TRAIN · 미리 학습",
    description: "가중치 변경 · 서비스 전",
    items: [
      ["train", "05", "모델 사전 학습"],
    ],
  },
  {
    id: "online",
    label: "실제 처리 · ONLINE",
    description: "질문이 들어올 때마다 반복",
    items: [
      ["generate", "06", "질문 인코딩 · 생성"],
      ["observe", "07", "어텐션 관찰"],
    ],
  },
] as const;

const navItems = [
  ...processGroups.flatMap((group) => group.items),
  ["remember", "✓", "개발 흐름 정리"],
] as const;

const sourcePrefix = ["하늘에", "", "보이면", "뭐가", "생각나"];
const specialTokens = [
  ["<pad>", "0", "길이를 맞추는 빈칸"],
  ["<sos>", "1", "답이 시작된다는 신호"],
  ["<eos>", "2", "답이 끝났다는 신호"],
  ["<unk>", "3", "단어장에 없는 낯선 말"],
];

const partTabs = [
  {
    id: "position",
    short: "위치",
    title: "단어마다 자리표를 붙여요",
    analogy: "같은 단어도 몇 번째 자리에 있느냐에 따라 다른 파도 무늬를 받습니다.",
    tech: "고정 sin/cos 표 (1, S, 64)를 임베딩에 더합니다.",
    shape: "(1, 5, 64)",
  },
  {
    id: "qkv",
    short: "Q · K · V",
    title: "검색어와 색인을 맞춰 봐요",
    analogy: "Q는 찾는 말, K는 책의 색인, V는 책 안의 실제 내용입니다.",
    tech: "QKᵀ/√16 → softmax → V의 가중합 순서입니다.",
    shape: "(1, 4, 5, 5)",
  },
  {
    id: "heads",
    short: "4개 헤드",
    title: "카메라 네 대로 동시에 봐요",
    analogy: "64칸을 16칸씩 나눠 네 관점으로 본 뒤 다시 합칩니다.",
    tech: "(B,S,64) → (B,4,S,16) → (B,S,64)",
    shape: "(1, 4, 5, 16)",
  },
  {
    id: "ffn",
    short: "FFN",
    title: "각 단어가 혼자 곱씹어요",
    analogy: "어텐션으로 들은 이야기를 단어마다 같은 작은 신경망에서 정리합니다.",
    tech: "위치별 독립 2층 신경망: Linear 64→256→64",
    shape: "(1, 5, 64)",
  },
] as const;

const lossPoints = [
  { epoch: 1, loss: 48.8795 },
  { epoch: 100, loss: 1.8443 },
  { epoch: 200, loss: 0.1063 },
  { epoch: 300, loss: 0.066 },
  { epoch: 400, loss: 0.0066 },
];

const codeSnippets = {
  data: `DATA = [
    ("하늘에 먹구름이 보이면 뭐가 생각나", "비가 올 것 같아"),
    ("하늘에 별이 보이면 뭐가 생각나", "밤이 깊었나 봐"),
    ("하늘에 해가 보이면 뭐가 생각나", "아침이 밝았구나"),
    # ... 모두 6쌍
]`,
  tokens: `def tokenize(sentence: str):
    return sentence.strip().split()

src = [torch.tensor(src_vocab.encode(q, add_special=False))
       for q, _ in pairs]
tgt = [torch.tensor(tgt_vocab.encode(a, add_special=True))
       for _, a in pairs]

src_ids = pad_sequence(src, batch_first=True,
                       padding_value=src_vocab.pad_id)`,
  encoder: `class EncoderLayer(nn.Module):
    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class Encoder(nn.Module):
    def forward(self, src, src_mask):
        x = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x`,
  decoder: `class DecoderLayer(nn.Module):
    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(
            x + self.dropout(self.self_attn(x, x, x, tgt_mask))
        )
        x = self.norm2(
            x + self.dropout(
                self.cross_attn(x, enc_out, enc_out, src_mask)
            )
        )
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x`,
  mask: `def make_causal_mask(seq_len, device):
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device)
    ).bool()
    return mask.unsqueeze(0).unsqueeze(0)

def make_decoder_mask(tgt, pad_id):
    return make_padding_mask(tgt, pad_id) & \\
           make_causal_mask(tgt.size(1), tgt.device)`,
  train: `dec_in, dec_tgt = tgt_ids[:, :-1], tgt_ids[:, 1:]
model.train()

for ep in range(1, epochs + 1):
    tgt_mask = make_decoder_mask(dec_in, tgt_vocab.pad_id)
    logits = model(src_ids, dec_in, src_mask, tgt_mask)
    loss = crit(
        logits.reshape(-1, logits.size(-1)),
        dec_tgt.reshape(-1)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()`,
  generate: `@torch.no_grad()
def answer(model, src_vocab, tgt_vocab, question, max_len=20):
    model.eval()
    src_ids = torch.tensor([
        src_vocab.encode(question, add_special=False)
    ])
    src_mask = make_padding_mask(src_ids, src_vocab.pad_id)
    enc = model.encode(src_ids, src_mask)  # 질문은 한 번만
    gen = [tgt_vocab.sos_id]

    for _ in range(max_len):
        tgt = torch.tensor([gen])
        tgt_mask = make_causal_mask(tgt.size(1), tgt.device)
        logits = model.decode_step(tgt, enc, tgt_mask, src_mask)
        nxt = logits[0, -1].argmax().item()
        gen.append(nxt)
        if nxt == tgt_vocab.eos_id:
            break`,
  attention: `model.decode_step(tgt, enc, tgt_mask, src_mask)

cross = (
    model.decoder.layers[-1]
    .cross_attn.last_attn_weights[0]
    .mean(0)
)  # (tgt_len, src_len)

weights = cross[:len(a_tokens), :len(q_tokens)].numpy()
ax.imshow(weights, aspect="auto", cmap="viridis")`,
  main: `src_vocab = Vocab([q for q, _ in DATA])
tgt_vocab = Vocab([a for _, a in DATA])

# OFFLINE: 서비스 전에 학습
model, history = train(DATA, src_vocab, tgt_vocab)

# ONLINE: 질문마다 추론
main_q = "하늘에 먹구름이 보이면 뭐가 생각나"
steps, ans, cross = answer(
    model, src_vocab, tgt_vocab, main_q
)
plot_attention(main_q, ans, cross, out_path)`,
};

function WeatherPicker({
  selected,
  onSelect,
  compact = false,
}: {
  selected: number;
  onSelect: (index: number) => void;
  compact?: boolean;
}) {
  return (
    <div className={`weather-picker ${compact ? "compact" : ""}`} aria-label="학습 예시 선택">
      {samples.map((sample, index) => (
        <button
          key={sample.key}
          type="button"
          className={selected === index ? "active" : ""}
          aria-pressed={selected === index}
          onClick={() => onSelect(index)}
          style={{ "--sample-accent": sample.accent } as React.CSSProperties}
        >
          <span className="picker-dot" aria-hidden="true" />
          {sample.label}
        </button>
      ))}
    </div>
  );
}

function ShapePill({ children }: { children: React.ReactNode }) {
  return (
    <span className="shape-pill" title="텐서의 모양">
      {children}
    </span>
  );
}

function SectionHeading({
  eyebrow,
  title,
  copy,
}: {
  eyebrow: string;
  title: string;
  copy: string;
}) {
  return (
    <div className="section-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{copy}</p>
    </div>
  );
}

function CodePanel({
  file,
  lines,
  title,
  explanation,
  code,
  dark = false,
}: {
  file: "demo.py" | "mini_transformer.py";
  lines: string;
  title: string;
  explanation: string;
  code: string;
  dark?: boolean;
}) {
  return (
    <div className={`code-panel ${dark ? "dark-code" : ""}`}>
      <div className="code-panel-copy">
        <p className="code-kicker">실제 코드 · {file}:{lines}</p>
        <h3>{title}</h3>
        <p>{explanation}</p>
        <a href={`/${file}`} download>
          {file} 전체 받기 <span aria-hidden="true">↓</span>
        </a>
      </div>
      <div className="code-window">
        <div className="code-window-bar">
          <span aria-hidden="true"><i /><i /><i /></span>
          <strong>{file}</strong>
          <small>Python</small>
        </div>
        <pre tabIndex={0} aria-label={`${file} ${lines}줄 코드`}><code>{code}</code></pre>
      </div>
    </div>
  );
}

export default function Home() {
  const [selected, setSelected] = useState(0);
  const [activeSection, setActiveSection] = useState("data");
  const [tokenStage, setTokenStage] = useState(0);
  const [partIndex, setPartIndex] = useState(0);
  const [position, setPosition] = useState(1);
  const [maskRow, setMaskRow] = useState(2);
  const [lossIndex, setLossIndex] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [generationStep, setGenerationStep] = useState(0);
  const sample = samples[selected];
  const questionTokens = useMemo(
    () => sourcePrefix.map((token, index) => (index === 1 ? sample.subject : token)),
    [sample],
  );
  const generated = sample.answerTokens.slice(0, generationStep);

  useEffect(() => {
    setGenerationStep(0);
    setTokenStage(0);
  }, [selected]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActiveSection(visible.target.id);
      },
      { rootMargin: "-25% 0px -55% 0px", threshold: [0.1, 0.35, 0.6] },
    );
    navItems.forEach(([id]) => {
      const section = document.getElementById(id);
      if (section) observer.observe(section);
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isTraining) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setIsTraining(false);
      return;
    }
    const timer = window.setInterval(() => {
      setLossIndex((current) => {
        if (current >= lossPoints.length - 1) {
          setIsTraining(false);
          return current;
        }
        return current + 1;
      });
    }, 650);
    return () => window.clearInterval(timer);
  }, [isTraining]);

  const heatValues = sample.answerTokens.map((_, rowIndex) =>
    questionTokens.map((__, colIndex) => {
      if (colIndex === 1) return Math.max(0.36, 0.58 - rowIndex * 0.07);
      const base = [0.09, 0.12, 0.08, 0.07, 0.06][colIndex];
      return Math.min(0.28, base + rowIndex * 0.025);
    }),
  );

  const nextGeneration = () => {
    setGenerationStep((current) =>
      current >= sample.answerTokens.length + 1 ? 0 : current + 1,
    );
  };

  return (
    <main style={{ "--active-accent": sample.accent } as React.CSSProperties}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="처음으로">
          <span className="brand-mark" aria-hidden="true">T</span>
          <span>Transformer Journey</span>
        </a>
        <WeatherPicker selected={selected} onSelect={setSelected} compact />
        <a className="notebook-link" href="/example-explain.ipynb" download>
          노트북 받기 <span aria-hidden="true">↓</span>
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="kicker"><span /> 6쌍의 문장으로 시작하는 미니 실험</div>
          <h1>먹구름을 보고<br /><em>‘비’</em>를 떠올리기까지</h1>
          <p>
            문장이 숫자가 되고, 중요한 말에 주목하고, 답을 한 단어씩 만드는
            전 과정을 직접 움직여 보세요.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#data">여행 시작하기 <span aria-hidden="true">↓</span></a>
            <a className="button text" href="#encode">전체 구조 먼저 보기 <span aria-hidden="true">→</span></a>
          </div>
          <dl className="hero-stats">
            <div><dt>64</dt><dd>단어 벡터 칸</dd></div>
            <div><dt>4</dt><dd>서로 다른 시선</dd></div>
            <div><dt>2 + 2</dt><dd>인코더 · 디코더 층</dd></div>
          </dl>
        </div>
        <figure className="hero-visual">
          <img
            src="/og.png"
            alt="먹구름 정보가 네 갈래 어텐션 경로를 지나 빗방울 답으로 바뀌는 교육용 그림"
          />
          <figcaption>
            <span className="pulse-dot" aria-hidden="true" />
            한 토큰의 여행을 왼쪽에서 오른쪽으로 따라가세요
          </figcaption>
        </figure>
      </section>

      <section className="developer-orientation" aria-labelledby="development-map-title">
        <div className="orientation-heading">
          <p className="eyebrow">먼저, 세 종류의 일을 나눠 볼게요</p>
          <h2 id="development-map-title">모델을 만들고, 미리 학습한 뒤, 질문마다 답해요</h2>
          <p>
            이 데모는 한 번 실행 안에서 학습과 추론을 차례로 수행하지만, 개발·학습·추론의 역할은
            완전히 다릅니다.
          </p>
        </div>
        <div className="runtime-split">
          <article className="runtime-card develop-card">
            <div className="runtime-label"><span aria-hidden="true">◇</span> DEVELOP · 모델 만들기</div>
            <h3>무엇을 배울지 설계</h3>
            <ol>
              <li><span>01</span>데이터와 단어장 준비</li>
              <li><span>02</span>Transformer 부품 구현</li>
            </ol>
            <p><code>mini_transformer.py</code>와 <code>demo.py</code>를 작성하는 단계</p>
          </article>
          <div className="runtime-bridge" aria-hidden="true">
            <span>코드로<br />실행</span>
            <i>→</i>
          </div>
          <article className="runtime-card offline-card">
            <div className="runtime-label"><span aria-hidden="true">○</span> OFFLINE TRAIN · 미리 학습</div>
            <h3>답하기 전에 가중치 학습</h3>
            <ol>
              <li><span>01</span><code>model.train()</code></li>
              <li><span>02</span>6쌍 × 400 epoch</li>
              <li><span>03</span><code>loss.backward()</code> · <code>opt.step()</code></li>
            </ol>
            <p><strong>가중치가 바뀝니다.</strong> 실제 서비스라면 학습이 끝난 모델을 저장·배포합니다.</p>
          </article>
          <div className="runtime-bridge weight-bridge" aria-hidden="true">
            <span>학습된<br />가중치 전달</span>
            <i>→</i>
          </div>
          <article className="runtime-card online-card">
            <div className="runtime-label"><span aria-hidden="true">●</span> ONLINE · 실제 처리</div>
            <h3>질문이 올 때마다 추론</h3>
            <ol>
              <li><span>01</span><code>model.eval()</code> · <code>@torch.no_grad()</code></li>
              <li><span>02</span>질문 1개를 한 번 인코딩</li>
              <li><span>03</span><code>argmax</code>로 답 생성</li>
            </ol>
            <p><strong>가중치는 고정됩니다.</strong> 학습 없이 답과 attention만 계산합니다.</p>
          </article>
        </div>
        <p className="demo-accuracy-note">
          <strong>이 미니 demo.py는 모델을 저장하지 않습니다.</strong>
          실행할 때 먼저 400 epoch를 학습한 뒤 같은 실행 안에서 바로 추론합니다.
        </p>
        <div className="runtime-table-wrap">
          <table>
            <caption>사전 학습과 실제 추론 비교</caption>
            <thead><tr><th>구분</th><th>입력</th><th>PyTorch 모드</th><th>반복</th><th>가중치</th><th>결과</th></tr></thead>
            <tbody>
              <tr><th>OFFLINE</th><td>질문·답 6쌍</td><td><code>train()</code></td><td>400 epoch</td><td>변경</td><td>학습된 모델 · loss</td></tr>
              <tr><th>ONLINE</th><td>질문 1개</td><td><code>eval() + no_grad()</code></td><td>&lt;eos&gt;까지</td><td>고정</td><td>답 · attention</td></tr>
            </tbody>
          </table>
        </div>
        <div className="file-role-grid">
          <article>
            <span className="file-icon" aria-hidden="true">M</span>
            <div>
              <small>MODEL CORE</small>
              <h3>mini_transformer.py</h3>
              <p>어텐션, 인코더, 디코더, 마스크와 전체 Transformer 클래스</p>
            </div>
            <a href="/mini_transformer.py" download aria-label="mini_transformer.py 내려받기">↓</a>
          </article>
          <article>
            <span className="file-icon demo-icon" aria-hidden="true">D</span>
            <div>
              <small>RUNNER</small>
              <h3>demo.py</h3>
              <p>데이터·단어장·배치·학습 반복·greedy 생성·히트맵 저장</p>
            </div>
            <a href="/demo.py" download aria-label="demo.py 내려받기">↓</a>
          </article>
        </div>
      </section>

      <div className="journey-layout">
        <aside className="progress-rail" aria-label="학습 단계">
          <p>PROCESS</p>
          <nav>
            {processGroups.map((group) => (
              <div className={`process-group ${group.id}`} key={group.id}>
                <div className="process-group-heading">
                  <strong>{group.label}</strong>
                  <small>{group.description}</small>
                </div>
                {group.items.map(([id, number, label]) => (
                  <a
                    key={id}
                    href={`#${id}`}
                    className={activeSection === id ? "active" : ""}
                    aria-current={activeSection === id ? "step" : undefined}
                  >
                    <span>{number}</span>
                    {label}
                  </a>
                ))}
              </div>
            ))}
            <a
              href="#remember"
              className={`process-summary-link ${activeSection === "remember" ? "active" : ""}`}
              aria-current={activeSection === "remember" ? "step" : undefined}
            >
              <span>✓</span>개발 흐름 정리
            </a>
          </nav>
          <div className="shape-tracker">
            <small>지금 흐르는 모양</small>
            <strong>
              {activeSection === "data" && "(1, 5)"}
              {activeSection === "tokens" && "(1, 5)"}
              {activeSection === "encode" && "(1, 5, 64)"}
              {activeSection === "decode" && "(1, T, 64)"}
              {activeSection === "train" && "(6, 5, 18)"}
              {activeSection === "generate" && "(1, T, 18)"}
              {activeSection === "observe" && "(T, 5)"}
              {activeSection === "remember" && "완료!"}
            </strong>
          </div>
        </aside>

        <div className="content-flow">
          <section className="lesson-section" id="data">
            <SectionHeading
              eyebrow="DEVELOP 01 · 학습 데이터 설계"
              title="정답을 가르는 단서는 단 하나"
              copy="서비스를 만들기 전, 먼저 모델이 배울 질문과 답을 준비합니다. 질문의 틀은 그대로 두고 대상만 바꿨습니다."
            />
            <WeatherPicker selected={selected} onSelect={setSelected} />
            <div className="sentence-stage">
              <div className="sentence-question" aria-live="polite">
                하늘에 <mark>{sample.subject}</mark> 보이면 뭐가 생각나
              </div>
              <div className="flow-arrow" aria-hidden="true">→</div>
              <div className="sentence-answer">{sample.answer}</div>
            </div>
            <div className="insight-strip">
              <span className="icon-badge" aria-hidden="true">!</span>
              <p><strong>관찰 포인트</strong> 정답을 바꾸는 유일한 단서는 <mark>{sample.subject}</mark>입니다.</p>
            </div>
            <CodePanel
              file="demo.py"
              lines="42–49"
              title="학습 예시는 Python 튜플로 시작합니다"
              explanation="실제 demo.py도 같은 6쌍을 DATA에 선언합니다. 이 작은 데이터는 일반화보다 어텐션 구조를 또렷하게 보여주기 위한 교육용 설계입니다."
              code={codeSnippets.data}
            />
          </section>

          <section className="lesson-section" id="tokens">
            <SectionHeading
              eyebrow="DEVELOP 02 · 단어장과 배치 준비"
              title="컴퓨터가 읽을 수 있게 잘게 쪼개요"
              copy="학습 전에 문장을 토큰 ID로 바꾸고, 길이를 맞춘 배치 텐서를 만듭니다. 이 준비도 질문마다 반복하는 과정이 아닙니다."
            />
            <div className="token-workbench">
              <div className="token-controls">
                <button
                  className={tokenStage === 0 ? "active" : ""}
                  type="button"
                  onClick={() => setTokenStage(0)}
                >
                  1. 문장
                </button>
                <button
                  className={tokenStage === 1 ? "active" : ""}
                  type="button"
                  onClick={() => setTokenStage(1)}
                >
                  2. 토큰
                </button>
                <button
                  className={tokenStage === 2 ? "active" : ""}
                  type="button"
                  onClick={() => setTokenStage(2)}
                >
                  3. 번호
                </button>
              </div>
              <div className="token-output" aria-live="polite">
                {tokenStage === 0 && (
                  <p>“하늘에 {sample.subject} 보이면 뭐가 생각나”</p>
                )}
                {tokenStage >= 1 && questionTokens.map((token, index) => (
                  <div className={`token-stack ${index === 1 ? "key-token" : ""}`} key={`${token}-${index}`}>
                    <span className="token-chip">{token}</span>
                    {tokenStage === 2 && <span className="token-id">{sample.sourceIds[index]}</span>}
                  </div>
                ))}
              </div>
              <div className="shape-line">
                <span>문장 1개</span><i aria-hidden="true" /><span>토큰 5개</span><i aria-hidden="true" />
                <ShapePill>(1, 5)</ShapePill>
              </div>
            </div>
            <div className="special-grid">
              {specialTokens.map(([token, id, description]) => (
                <article key={token}>
                  <code>{token}</code>
                  <span>{id}</span>
                  <p>{description}</p>
                </article>
              ))}
            </div>
            <CodePanel
              file="demo.py"
              lines="52–85, 105–111"
              title="토큰화한 문장을 배치 텐서로 묶습니다"
              explanation="질문과 답은 서로 다른 단어장을 쓰고, 답에는 <sos>와 <eos>를 붙입니다. pad_sequence가 길이를 맞춰 (B, L) 텐서를 만듭니다."
              code={codeSnippets.tokens}
            />
          </section>

          <section className="lesson-section dark-section" id="encode">
            <SectionHeading
              eyebrow="DEVELOP 03 · 인코더 구현"
              title="질문을 이해하는 부품을 개발해요"
              copy="이 단계는 실제 질문을 처리하는 장면이 아니라, 그 처리를 담당할 Encoder 클래스를 미리 구현하는 과정입니다."
            />
            <div className="assembly-map" aria-label="인코더 처리 순서">
              <div><small>입력</small><strong>단어 ID</strong><ShapePill>(1, 5)</ShapePill></div>
              <span>→</span>
              <div><small>의미 + 위치</small><strong>Embedding</strong><ShapePill>(1, 5, 64)</ShapePill></div>
              <span>→</span>
              <div className="stacked"><small>2층 반복</small><strong>Attention + FFN</strong><ShapePill>(1, 5, 64)</ShapePill></div>
              <span>→</span>
              <div><small>질문의 기억</small><strong>Memory</strong><ShapePill>(1, 5, 64)</ShapePill></div>
            </div>
            <div className="parts-workbench">
              <div className="part-tabs" role="tablist" aria-label="인코더 부품">
                {partTabs.map((part, index) => (
                  <button
                    key={part.id}
                    role="tab"
                    aria-selected={partIndex === index}
                    type="button"
                    onClick={() => setPartIndex(index)}
                  >
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    {part.short}
                  </button>
                ))}
              </div>
              <div className="part-panel" role="tabpanel">
                <div className="part-copy">
                  <p className="analogy-label">쉬운 비유</p>
                  <h3>{partTabs[partIndex].title}</h3>
                  <p>{partTabs[partIndex].analogy}</p>
                  <div className="computer-note">
                    <span>컴퓨터 안에서는</span>
                    <code>{partTabs[partIndex].tech}</code>
                  </div>
                  <ShapePill>{partTabs[partIndex].shape}</ShapePill>
                </div>
                <div className={`part-visual part-${partTabs[partIndex].id}`}>
                  {partTabs[partIndex].id === "position" && (
                    <>
                      <div className="word-position">
                        {questionTokens.map((token, index) => (
                          <button
                            type="button"
                            key={token}
                            className={position === index ? "active" : ""}
                            onClick={() => setPosition(index)}
                            aria-label={`${index + 1}번째 단어 ${token}`}
                          >
                            <small>{index + 1}</small>{token}
                          </button>
                        ))}
                      </div>
                      <div className="waves" aria-label={`${position + 1}번째 위치의 사인 코사인 파도 예시`}>
                        {Array.from({ length: 16 }, (_, index) => (
                          <i
                            key={index}
                            style={{ height: `${28 + Math.abs(Math.sin((index + position) * 0.7)) * 58}%` }}
                          />
                        ))}
                      </div>
                      <p>{position + 1}번째 자리만의 파도 무늬</p>
                    </>
                  )}
                  {partTabs[partIndex].id === "qkv" && (
                    <div className="qkv-stage">
                      <div className="qkv-node q">Q<small>검색어</small></div>
                      <div className="qkv-lines" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                      <div className="keys-row">
                        {questionTokens.map((token, index) => (
                          <span key={token} className={index === 1 ? "active" : ""}>
                            <b>K</b>{token}<small>{index === 1 ? "가장 가까움" : "색인"}</small>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {partTabs[partIndex].id === "heads" && (
                    <div className="heads-stage">
                      {[["01", "대상"], ["02", "문맥"], ["03", "순서"], ["04", "말투"]].map(([number, label], index) => (
                        <div key={number} className="head-card">
                          <span>{number}</span><strong>{label}</strong>
                          <div className="head-meter"><i style={{ width: `${[94, 61, 48, 73][index]}%` }} /></div>
                          <small>16칸 관점</small>
                        </div>
                      ))}
                    </div>
                  )}
                  {partTabs[partIndex].id === "ffn" && (
                    <div className="ffn-stage">
                      <div><strong>64</strong><small>들어온 생각</small></div>
                      <span aria-hidden="true">→</span>
                      <div className="wide"><strong>256</strong><small>넓게 곱씹기</small></div>
                      <span aria-hidden="true">→</span>
                      <div><strong>64</strong><small>다시 정리</small></div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <CodePanel
              file="mini_transformer.py"
              lines="103–118, 141–157"
              title="Self-Attention과 FFN을 한 층으로 묶습니다"
              explanation="EncoderLayer는 질문 안의 단어끼리 서로 보게 하고, 잔차 연결과 LayerNorm으로 안정화합니다. Encoder는 이 층을 두 번 반복합니다."
              code={codeSnippets.encoder}
              dark
            />
          </section>

          <section className="lesson-section" id="decode">
            <SectionHeading
              eyebrow="DEVELOP 04 · 디코더와 마스크 구현"
              title="답을 만들 부품은 두 번 주목해요"
              copy="학습과 추론에서 함께 쓸 DecoderLayer와 미래 가림 함수를 미리 개발합니다. 질문을 받으면 이 코드가 실제로 실행됩니다."
            />
            <div className="decoder-story">
              <article>
                <span className="step-number">1</span>
                <div>
                  <small>Masked self-attention</small>
                  <h3>내가 지금까지 쓴 답 보기</h3>
                  <p>&lt;sos&gt;와 앞말만 볼 수 있어요.</p>
                </div>
              </article>
              <span aria-hidden="true">→</span>
              <article className="accent-card">
                <span className="step-number">2</span>
                <div>
                  <small>Cross-attention</small>
                  <h3>질문의 <mark>{sample.subject}</mark> 곁눈질</h3>
                  <p>Q는 답, K와 V는 질문의 기억입니다.</p>
                </div>
              </article>
              <span aria-hidden="true">→</span>
              <article>
                <span className="step-number">3</span>
                <div>
                  <small>Feed-forward</small>
                  <h3>다음 단어 후보 정리</h3>
                  <p>18개 답 단어에 점수를 매겨요.</p>
                </div>
              </article>
            </div>
            <div className="mask-lab">
              <div className="mask-copy">
                <p className="eyebrow">미래 가리개</p>
                <h3>{maskRow + 1}번째 답칸이 볼 수 있는 범위</h3>
                <p>끝말잇기처럼 이미 나온 말까지만 볼 수 있습니다. 정답을 미리 훔쳐보지 못하게 하는 규칙이에요.</p>
                <div className="mask-row-picker" aria-label="살펴볼 답 위치">
                  {[0, 1, 2, 3, 4].map((row) => (
                    <button
                      type="button"
                      key={row}
                      className={maskRow === row ? "active" : ""}
                      onClick={() => setMaskRow(row)}
                    >
                      {row + 1}
                    </button>
                  ))}
                </div>
              </div>
              <div className="mask-grid" role="img" aria-label={`${maskRow + 1}번째 위치는 앞의 ${maskRow + 1}칸만 볼 수 있음`}>
                {Array.from({ length: 25 }, (_, index) => {
                  const row = Math.floor(index / 5);
                  const col = index % 5;
                  const visible = col <= row;
                  return (
                    <span
                      key={index}
                      className={`${visible ? "visible" : "hidden"} ${row === maskRow ? "current" : ""}`}
                    >
                      {visible ? "●" : "×"}
                    </span>
                  );
                })}
              </div>
            </div>
            <CodePanel
              file="mini_transformer.py"
              lines="121–138"
              title="Masked Self-Attention 다음에 질문을 봅니다"
              explanation="첫 번째 어텐션은 지금까지의 답만 보고, 두 번째 cross_attn은 Q=답, K/V=인코더 출력으로 질문을 참고합니다."
              code={codeSnippets.decoder}
            />
            <CodePanel
              file="mini_transformer.py"
              lines="207–215"
              title="하삼각 행렬로 미래 답을 가립니다"
              explanation="make_decoder_mask는 패딩 마스크와 causal mask를 AND로 합칩니다. 결과 shape은 (B, 1, T, T)입니다."
              code={codeSnippets.mask}
            />
          </section>

          <section className="lesson-section offline-training-section" id="train">
            <div className="phase-banner offline-banner">
              <span>OFFLINE</span>
              <strong>여기까지는 서비스 전에 미리 합니다</strong>
              <p>학습이 끝난 모델의 가중치를 실제 추론에 사용합니다.</p>
            </div>
            <SectionHeading
              eyebrow="OFFLINE 05 · 모델 사전 학습"
              title="질문을 받기 전에 미리 학습해요"
              copy="6개 문장을 한 묶음으로 400번 반복해 가중치를 만듭니다. 서비스에서 질문을 처리할 때는 이 학습 반복을 실행하지 않습니다."
            />
            <div className="simulation-note">
              <span>사전 계산된 학습 기록</span>
              웹에서는 loss 기록을 재생합니다. 실제 PyTorch 학습은 <code>demo.py</code>의 <code>train()</code>이 실행합니다.
            </div>
            <div className="training-cycle">
              {[
                ["1", "예측", "다음 단어 점수 만들기"],
                ["2", "비교", "정답과 loss 계산하기"],
                ["3", "수정", "가중치를 한 걸음 고치기"],
              ].map(([number, title, copy], index) => (
                <div key={number}>
                  <span>{number}</span>
                  <strong>{title}</strong>
                  <small>{copy}</small>
                  {index < 2 && <i aria-hidden="true">→</i>}
                </div>
              ))}
            </div>
            <div className="loss-lab">
              <div className="loss-chart" aria-label={`epoch ${lossPoints[lossIndex].epoch}, loss ${lossPoints[lossIndex].loss}`}>
                <div className="chart-grid" aria-hidden="true" />
                <div className="chart-bars">
                  {lossPoints.map((point, index) => (
                    <button
                      type="button"
                      key={point.epoch}
                      className={index <= lossIndex ? "revealed" : ""}
                      onClick={() => setLossIndex(index)}
                      aria-label={`epoch ${point.epoch}, loss ${point.loss}`}
                    >
                      <i style={{ height: `${12 + Math.min(78, Math.sqrt(point.loss / 48.8795) * 78)}%` }} />
                      <span>{point.epoch}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="loss-readout" aria-live="polite">
                <p>epoch</p>
                <strong>{lossPoints[lossIndex].epoch}</strong>
                <p>loss</p>
                <strong className="loss-number">{lossPoints[lossIndex].loss.toFixed(4)}</strong>
                <div className="loss-actions">
                  <button type="button" onClick={() => setLossIndex((current) => Math.min(current + 1, lossPoints.length - 1))}>
                    다음 기록 보기
                  </button>
                  <button type="button" onClick={() => { setLossIndex(0); setIsTraining(true); }}>
                    {isTraining ? "재생 중…" : "학습 곡선 재생"}
                  </button>
                </div>
              </div>
            </div>
            <div className="fact-row">
              <span><b>배치</b> (6, 5)</span>
              <span><b>출력</b> (6, 5, 18)</span>
              <span><b>학습 파라미터</b> 235,520</span>
              <span><b>최종 loss</b> 0.0066</span>
            </div>
            <CodePanel
              file="demo.py"
              lines="114–138"
              title="teacher forcing으로 다음 단어를 미리 학습합니다"
              explanation="답 입력과 정답을 한 칸 어긋나게 만들고 forward → loss → backward → optimizer.step 순서로 가중치를 갱신합니다."
              code={codeSnippets.train}
            />
          </section>

          <section className="lesson-section dark-section generation-section" id="generate">
            <div className="phase-banner online-banner">
              <span>ONLINE</span>
              <strong>여기부터 질문이 들어올 때 실행합니다</strong>
              <p>학습은 끄고, 저장된 가중치로 답만 계산합니다.</p>
            </div>
            <SectionHeading
              eyebrow="ONLINE 01 · 질문 인코딩과 생성"
              title="학습된 모델이 답을 한 칸씩 써요"
              copy="질문을 한 번 인코딩한 뒤, 매 순간 가장 점수가 높은 단어 하나를 골라 다음 입력에 붙입니다."
            />
            <div className="online-call-path" aria-label="질문이 들어온 뒤 실제 호출 순서">
              {[
                ["1", "tokenize", "새 질문을 단어로"],
                ["2", "Vocab.encode", "ID로 변환"],
                ["3", "padding mask", "빈칸 가리기"],
                ["4", "model.encode", "질문은 한 번"],
                ["5", "decode_step", "답은 반복"],
              ].map(([number, call, copy], index) => (
                <div key={call} className={index === 4 ? "repeat-call" : ""}>
                  <span>{number}</span><code>{call}</code><small>{copy}</small>
                </div>
              ))}
            </div>
            <div className="simulation-note dark-simulation-note">
              <span>저장된 예시 결과 재생</span>
              아래 버튼은 노트북과 demo.py에서 나온 생성 순서를 브라우저에서 설명용으로 재생합니다.
            </div>
            <div className="generation-lab">
              <div className="generation-question">
                <small>인코더가 한 번만 읽은 질문</small>
                <p>하늘에 <mark>{sample.subject}</mark> 보이면 뭐가 생각나</p>
              </div>
              <div className="generation-track" aria-live="polite">
                <span className="special">&lt;sos&gt;</span>
                {sample.answerTokens.map((token, index) => (
                  <span
                    key={`${sample.key}-${token}`}
                    className={index < generationStep ? "generated" : "future"}
                  >
                    {index < generationStep ? token : "미래"}
                  </span>
                ))}
                <span className={generationStep > sample.answerTokens.length ? "generated special" : "future"}>
                  {generationStep > sample.answerTokens.length ? "<eos>" : "끝"}
                </span>
              </div>
              <div className="generation-result">
                <div>
                  <small>지금까지 만든 답</small>
                  <strong>{generated.length ? generated.join(" ") : "아직 빈 문장"}</strong>
                </div>
                <button type="button" onClick={nextGeneration}>
                  {generationStep > sample.answerTokens.length ? "처음부터" : "다음 단어 만들기"}
                  <span aria-hidden="true">→</span>
                </button>
              </div>
            </div>
            <div className="logits-note">
              <code>(1, T, 64)</code><span aria-hidden="true">→</span><code>Linear</code>
              <span aria-hidden="true">→</span><code>(1, T, 18)</code>
              <p>18개 후보 중 마지막 칸의 최고 점수를 고르는 greedy decoding</p>
            </div>
            <CodePanel
              file="demo.py"
              lines="142–166"
              title="eval 모드에서 질문은 한 번, 답은 반복 생성합니다"
              explanation="@torch.no_grad()와 model.eval()이 학습을 끕니다. encode 결과는 재사용하고, decode_step의 마지막 logits에서 argmax를 고릅니다."
              code={codeSnippets.generate}
              dark
            />
          </section>

          <section className="lesson-section" id="observe">
            <SectionHeading
              eyebrow="ONLINE 02 · 어텐션 관찰"
              title="밝을수록 그 질문 단어를 더 많이 봤어요"
              copy="마지막 디코더 층의 크로스 어텐션을 네 헤드 평균으로 모았습니다. 선택한 대상 열이 가장 밝아지는 패턴을 확인해 보세요."
            />
            <WeatherPicker selected={selected} onSelect={setSelected} />
            <div className="attention-layout">
              <div className="heatmap-wrap">
                <div
                  className="heatmap"
                  style={{ gridTemplateColumns: `92px repeat(${questionTokens.length}, minmax(76px, 1fr))` }}
                >
                  <span className="corner">답 ↓ / 질문 →</span>
                  {questionTokens.map((token, index) => (
                    <strong key={token} className={index === 1 ? "focus-label" : ""}>{token}</strong>
                  ))}
                  {sample.answerTokens.map((answerToken, rowIndex) => (
                    <div className="heat-row" key={answerToken} style={{ display: "contents" }}>
                      <strong>{answerToken}</strong>
                      {heatValues[rowIndex].map((value, colIndex) => (
                        <button
                          type="button"
                          key={`${answerToken}-${questionTokens[colIndex]}`}
                          style={{ "--heat": value } as React.CSSProperties}
                          aria-label={`답 ${answerToken}가 질문 ${questionTokens[colIndex]}를 ${Math.round(value * 100)}퍼센트 주목하는 개념 예시`}
                          title={`${answerToken} → ${questionTokens[colIndex]} · ${Math.round(value * 100)}%`}
                        >
                          <span>{Math.round(value * 100)}</span>
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
                <div className="heat-legend"><span>덜 봄</span><i /><span>더 봄</span></div>
              </div>
              <div className="attention-summary">
                <p className="eyebrow">무슨 일이 일어났나요?</p>
                <div className="spotlight-word">{sample.subject}</div>
                <p>답의 단어들이 공통 문장 틀보다 <strong>바뀐 대상 단어</strong>를 가장 강하게 바라봅니다.</p>
                <div className="attention-rank">
                  <span>주목 1위</span><strong>{sample.subject}</strong>
                  <span>생성한 답</span><strong>{sample.answer}</strong>
                </div>
                <small>표의 값은 상호작용을 위한 개념 시각화입니다. 노트북의 실제 실행도 모든 헤드에서 먹구름 열이 1위였습니다.</small>
              </div>
            </div>
            <CodePanel
              file="demo.py"
              lines="161–191"
              title="마지막 cross-attention을 꺼내 히트맵으로 저장합니다"
              explanation="마지막 디코더 층의 4개 헤드를 평균해 (답 길이, 질문 길이) 표를 만들고, matplotlib의 imshow로 attention_heatmap.png를 저장합니다."
              code={codeSnippets.attention}
            />
          </section>

          <section className="lesson-section final-section" id="remember">
            <SectionHeading
              eyebrow="정리 · 개발과 실행을 연결하기"
              title="두 파일을 이렇게 연결하면 완성돼요"
              copy="구조를 구현한 모듈을 demo.py에서 불러오고, 사전 학습과 실제 추론의 경계를 분명히 나눕니다."
            />
            <div className="memory-cards">
              <article>
                <span className="memory-icon underline-icon" aria-hidden="true">A</span>
                <p className="eyebrow">ATTENTION</p>
                <h3>중요한 말에<br />밑줄 긋기</h3>
                <p>답에 필요한 질문 단어를 골라 더 많이 가져옵니다.</p>
              </article>
              <article>
                <span className="memory-icon mask-icon" aria-hidden="true">×</span>
                <p className="eyebrow">MASKING</p>
                <h3>미래를 못 보는<br />끝말잇기</h3>
                <p>이미 만든 답만 보고 바로 다음 단어를 고릅니다.</p>
              </article>
              <article>
                <span className="memory-icon cross-icon" aria-hidden="true">↗</span>
                <p className="eyebrow">CROSS-ATTENTION</p>
                <h3>답을 쓰며<br />질문 곁눈질</h3>
                <p>디코더가 인코더의 질문 기억을 필요할 때 꺼냅니다.</p>
              </article>
            </div>
            <CodePanel
              file="demo.py"
              lines="194–236"
              title="main 함수가 OFFLINE과 ONLINE을 순서대로 연결합니다"
              explanation="현재 데모는 실행할 때마다 먼저 학습한 뒤 곧바로 추론합니다. 실제 서비스에서는 train 결과를 저장하고, 요청 처리 코드에서는 answer만 호출하도록 분리합니다."
              code={codeSnippets.main}
            />
            <div className="final-cta">
              <div>
                <p className="eyebrow">이제 직접 실행할 차례</p>
                <h2>노트북의 셀을 위에서 아래로 따라가 보세요.</h2>
                <p>방금 본 순서와 같은 모양이 코드 속에서 그대로 이어집니다.</p>
              </div>
              <a className="button primary" href="/example-explain.ipynb" download>
                노트북 내려받기 <span aria-hidden="true">↓</span>
              </a>
            </div>
          </section>
        </div>
      </div>

      <footer>
        <div className="brand"><span className="brand-mark" aria-hidden="true">T</span><span>Transformer Journey</span></div>
        <p>hands-on/example-explain.ipynb의 처리 순서를 따라 만든 동적 학습 페이지</p>
        <a href="#top">맨 위로 ↑</a>
      </footer>
    </main>
  );
}
