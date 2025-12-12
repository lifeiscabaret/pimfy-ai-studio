'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense } from 'react';

// 내용을 보여주는 컴포넌트
function ResultContent() {
    const searchParams = useSearchParams();
    const router = useRouter();

    // URL에서 'img' 라는 이름으로 전달된 이미지 주소를 꺼냅니다.
    const imageUrl = searchParams.get('img');

    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-pink-50 p-4">
            <div className="w-full max-w-lg bg-white rounded-3xl shadow-xl overflow-hidden p-6 flex flex-col items-center">

                {/* 상단 제목 */}
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-6">
                    <span className="text-brand-pink">♥</span> 핌피 프로필 도착 <span className="text-brand-pink">♥</span>
                </h1>

                {/* 이미지 영역 */}
                <div className="w-full rounded-2xl overflow-hidden shadow-sm border border-gray-100 mb-8 bg-gray-50 min-h-[300px] flex items-center justify-center">
                    {imageUrl ? (
                        <img
                            src={imageUrl}
                            alt="공유된 프로필"
                            className="w-full h-auto object-contain"
                        />
                    ) : (
                        <p className="font-kyobo text-gray-400">이미지를 불러올 수 없어요 🥲</p>
                    )}
                </div>

                {/* 하단 문구 */}
                <p className="font-kyobo text-center text-gray-600 mb-6 leading-relaxed">
                    세상에 하나뿐인 우리 아이 AI 프로필!<br />
                    지금 바로 만들어보세요 🐾
                </p>

                {/* 나도 하러 가기 버튼 */}
                <button
                    onClick={() => router.push('/')}
                    className="font-kyobo w-full bg-brand-pink text-white text-xl py-4 rounded-full shadow-lg hover:bg-opacity-90 transition-transform transform hover:scale-105 active:scale-95"
                >
                    나도 만들러 가기 👉
                </button>

            </div>
        </div>
    );
}

// 메인 페이지 (Suspense로 감싸야 에러가 안 납니다)
export default function ResultPage() {
    return (
        <Suspense fallback={<div className="text-center p-10 font-kyobo">로딩 중...</div>}>
            <ResultContent />
        </Suspense>
    );
}