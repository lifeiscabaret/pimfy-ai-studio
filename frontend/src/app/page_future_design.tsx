'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function Home() {
  const router = useRouter();
  const [isHovered, setIsHovered] = useState(false);

  const handleStart = () => {
    router.push('/profile');
  };

  return (
    <div className="min-h-screen bg-pink-50 overflow-hidden relative">

      {/* ============================================================
          📱 1. 모바일 화면 (md:hidden - 모바일에서만 보임)
          - 심플한 Serif 텍스트 + 뼈다귀 버튼
      ============================================================ */}
      <div className="flex md:hidden flex-col items-center justify-center min-h-screen p-6 relative z-10">

        {/* 타이틀 (애니메이션: 위에서 아래로 부드럽게 등장) */}
        <div className="animate-fadeInDown mb-16 text-center">
          <h1 className="font-serif text-5xl text-gray-900 leading-tight tracking-wide">
            Pimfy<br />
            <span className="italic text-brand-pink">AI</span> Photo
          </h1>
          <p className="font-kyobo text-gray-500 mt-4 text-lg">
            우리 아이를 위한 단 하나의 프로필
          </p>
        </div>

        {/* 뼈다귀 버튼 (SVG) */}
        <button
          onClick={handleStart}
          className="group relative w-64 h-24 transition-transform active:scale-95"
        >
          {/* 뼈다귀 모양 SVG */}
          <svg viewBox="0 0 300 100" className="w-full h-full drop-shadow-md group-hover:drop-shadow-xl transition-all">
            {/* 뼈다귀 몸통 */}
            <path
              d="M30,30  Q10,30 10,50  Q10,70 30,70  L270,70  Q290,70 290,50  Q290,30 270,30  Z"
              fill="#FF8ba7"
            />
            {/* 왼쪽 관절 */}
            <circle cx="30" cy="35" r="20" fill="#FF8ba7" />
            <circle cx="30" cy="65" r="20" fill="#FF8ba7" />
            {/* 오른쪽 관절 */}
            <circle cx="270" cy="35" r="20" fill="#FF8ba7" />
            <circle cx="270" cy="65" r="20" fill="#FF8ba7" />
          </svg>

          {/* 버튼 텍스트 (정중앙 배치) */}
          <span className="absolute inset-0 flex items-center justify-center font-kyobo text-white text-2xl font-bold pt-1">
            촬영하러 가기 📸
          </span>
        </button>

        {/* 하단 장식 (발바닥) */}
        <div className="absolute bottom-10 opacity-20 flex gap-4">
          <span className="text-4xl rotate-12">🐾</span>
          <span className="text-4xl -rotate-12">🐾</span>
        </div>
      </div>


      {/* ============================================================
          💻 2. PC/태블릿 화면 (hidden md:flex - 큰 화면에서만 보임)
          - 기존 포토부스 디자인 유지
      ============================================================ */}
      <div className="hidden md:flex min-h-screen flex-col items-center justify-center p-8">
        {/* (기존 배경 장식들) */}
        <div className="absolute top-10 left-10 w-32 h-32 bg-yellow-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute top-10 right-10 w-32 h-32 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute bottom-10 left-20 w-32 h-32 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>

        {/* 메인 타이틀 */}
        <h1 className="font-bungee text-6xl md:text-8xl text-center text-gray-800 mb-4 z-10 tracking-wider drop-shadow-sm">
          PIMFY <span className="text-brand-pink">PHOTO</span>
        </h1>
        <p className="font-kyobo text-xl md:text-2xl text-gray-600 mb-12 z-10 bg-white/50 px-6 py-2 rounded-full backdrop-blur-sm">
          단 한 장의 사진으로 시작하는 우리 아이들의 프로필! 📸
        </p>

        {/* 포토부스 컨테이너 */}
        <div
          className="relative group cursor-pointer perspective-1000 z-10"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          onClick={handleStart}
        >
          {/* 부스 프레임 */}
          <div className={`
            relative w-[300px] h-[400px] bg-white border-8 border-gray-800 rounded-lg shadow-2xl
            transition-all duration-500 ease-in-out transform-gpu
            ${isHovered ? 'rotate-y-12 scale-105 shadow-pink-500/20' : 'rotate-0 scale-100'}
          `}>
            {/* 커튼 (왼쪽) */}
            <div className={`
              absolute top-0 left-0 w-1/2 h-full bg-brand-pink border-r-2 border-pink-600/20 origin-left transition-transform duration-700 ease-in-out z-20
              ${isHovered ? '-scale-x-0' : 'scale-x-100'}
            `}></div>

            {/* 커튼 (오른쪽) */}
            <div className={`
              absolute top-0 right-0 w-1/2 h-full bg-brand-pink border-l-2 border-pink-600/20 origin-right transition-transform duration-700 ease-in-out z-20
              ${isHovered ? '-scale-x-0' : 'scale-x-100'}
            `}></div>

            {/* 내부 (강아지 사진 & 버튼) */}
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-100 overflow-hidden">
              <div className="w-48 h-48 bg-white rounded-full flex items-center justify-center shadow-inner mb-6 overflow-hidden border-4 border-white">
                <span className="text-6xl animate-bounce">🐶</span>
              </div>

              <button className="font-kyobo bg-gray-800 text-white px-6 py-3 rounded-full text-xl shadow-lg hover:bg-black transition-colors flex items-center gap-2">
                <span>촬영하기</span>
                <span className="text-sm">CLICK!</span>
              </button>
            </div>

            {/* 조명 효과 */}
            <div className="absolute top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-yellow-400 rounded-full shadow-[0_0_15px_rgba(250,204,21,0.8)] z-30 animate-pulse"></div>
          </div>

          {/* 바닥 그림자 */}
          <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 w-40 h-4 bg-black/20 blur-md rounded-[100%] transition-all duration-500 group-hover:w-56 group-hover:bg-black/30"></div>
        </div>
      </div>
    </div>
  );
}