import type { Metadata } from "next";
import "./globals.css";

import { Bungee_Shade } from "next/font/google";
import localFont from "next/font/local";

// 타이틀용(영문) – 시안의 레터링 폰트
const bungee = Bungee_Shade({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-bungee",
});

// 본문/한글용 – 교보 손글씨 로컬 번들
const kyobo = localFont({
  src: "./fonts/KyoboHandwriting2021sjy.otf",
  weight: "400",
  style: "normal",
  variable: "--font-kyobo",
});

export const metadata: Metadata = {
  title: "PIMFY PHOTO",
  description: "단 한 장의 사진으로 시작하는 우리 아이들의 프로필!",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className={`${bungee.variable} ${kyobo.variable}`}>

      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
