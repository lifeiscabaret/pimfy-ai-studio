import type { Metadata } from "next";
import "./globals.css";
import Script from "next/script";

import { Bungee_Shade } from "next/font/google";
import localFont from "next/font/local";

const bungee = Bungee_Shade({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-bungee",
});

const kyobo = localFont({
  src: "./fonts/KyoboHandwriting2021sjy.otf",
  weight: "400",
  style: "normal",
  variable: "--font-kyobo",
});

export const metadata: Metadata = {
  title: "PIMFY PHOTO",
  description: "단 한 장의 사진으로 시작하는 우리 아이들의 프로필!",
  referrer: 'origin',
};

declare global {
  interface Window {
    Kakao: any;
  }
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className={`${bungee.variable} ${kyobo.variable}`}>
      <body className="antialiased">
        {children}

        <Script
          src="https://developers.kakao.com/sdk/js/kakao.min.js"
          strategy="afterInteractive"
        />

        <Script id="kakao-init" strategy="afterInteractive">
          {`
            if (window.Kakao && !window.Kakao.isInitialized()) {
              window.Kakao.init('592b68bdf6a6bf3da19b7a6d958723b1'); 
              console.log("카카오 초기화 완료!");
            }
          `}
        </Script>
      </body>
    </html>
  );
}