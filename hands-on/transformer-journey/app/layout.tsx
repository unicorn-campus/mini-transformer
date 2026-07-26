import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.includes("localhost") ? "http" : "https");
  const imageUrl = `${protocol}://${host}/og.png`;

  return {
    title: "먹구름에서 ‘비’까지 | Transformer Journey",
    description: "문장이 숫자가 되고 어텐션을 거쳐 답이 되는 트랜스포머의 처리 순서를 직접 움직이며 배워보세요.",
    icons: {
      icon: "/og.png",
      shortcut: "/og.png",
    },
    openGraph: {
      title: "먹구름에서 ‘비’까지",
      description: "한 토큰의 여행으로 이해하는 미니 트랜스포머",
      type: "website",
      locale: "ko_KR",
      images: [{ url: imageUrl, width: 1680, height: 945, alt: "먹구름에서 비까지 이어지는 트랜스포머 학습 여정" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "먹구름에서 ‘비’까지",
      description: "한 토큰의 여행으로 이해하는 미니 트랜스포머",
      images: [imageUrl],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
