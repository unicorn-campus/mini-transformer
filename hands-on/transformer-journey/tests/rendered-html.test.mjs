import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html", host: "localhost" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the transformer learning journey", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /먹구름에서 ‘비’까지/);
  assert.match(html, /Transformer Journey/);
  assert.match(html, /학습 데이터 설계/);
  assert.match(html, /DEVELOP · 모델 만들기/);
  assert.match(html, /OFFLINE TRAIN · 미리 학습/);
  assert.match(html, /ONLINE · 실제 처리/);
  assert.match(html, /질문을 받기 전에 미리 학습해요/);
  assert.match(html, /실제 코드/);
  assert.match(html, /demo\.py/);
  assert.match(html, /mini_transformer\.py/);
  assert.match(html, /og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/);
});

test("ships the generated image, notebook, and accessible interaction labels", async () => {
  const [page, layout] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    access(new URL("../public/og.png", import.meta.url)),
    access(new URL("../public/example-explain.ipynb", import.meta.url)),
    access(new URL("../public/demo.py", import.meta.url)),
    access(new URL("../public/mini_transformer.py", import.meta.url)),
  ]);

  assert.match(page, /aria-live="polite"/);
  assert.match(page, /prefers-reduced-motion/);
  assert.match(page, /WeatherPicker/);
  assert.match(page, /Cross-attention/);
  assert.match(page, /codeSnippets\.train/);
  assert.match(page, /model\.eval\(\)/);
  assert.match(page, /loss\.backward\(\)/);
  assert.match(layout, /x-forwarded-host/);
  assert.match(layout, /summary_large_image/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
