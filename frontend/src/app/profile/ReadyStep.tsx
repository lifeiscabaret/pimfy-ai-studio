'use client';

import { useState } from 'react';
import Image from 'next/image';

const IconKakao = () => (
    <svg viewBox="0 0 32 32" className="w-6 h-6">
        <path fill="currentColor" d="M16 4.64c-6.96 0-12.64 4.48-12.64 10.08 0 3.52 2.32 6.64 5.76 8.48l-.96 3.52.96-.08 3.2-2.24c1.2.32 2.48.56 3.68.56 6.96 0 12.64-4.48 12.64-10.24S22.96 4.64 16 4.64z" />
    </svg>
);
const IconInstagram = () => (
    <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect>
        <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
        <line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line>
    </svg>
);
const IconSave = () => (
    <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="7 10 12 15 17 10"></polyline>
        <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
);


interface ReadyStepProps {
    onRetry: () => void;
    onGoHome: () => void;
}

export default function ReadyStep({ onRetry, onGoHome }: ReadyStepProps) {
    const handleShareKakao = () => alert('카카오톡 공유 기능 구현 예정');
    const handleShareInsta = () => alert('인스타그램 공유 기능 구현 예정');
    const handleDownloadImage = () => alert('사진 저장 기능 구현 예정');

    return (
        <div className="w-full max-w-2xl flex flex-col items-center bg-white p-8 rounded-2xl shadow-lg">
            <h1 className="font-kyobo text-4xl mb-6">프로필 완성!</h1>

            <div className="w-full max-w-sm aspect-[3/4] bg-gray-200 rounded-lg mb-8 flex items-center justify-center">
                <p className="font-kyobo text-2xl text-gray-400">완성된 프로필 이미지</p>
            </div>

            <div className="flex items-center gap-6 mb-10">
                <button onClick={handleShareKakao} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                    <IconKakao />
                    <span className="font-kyobo text-sm">카톡 보내기</span>
                </button>
                <button onClick={handleShareInsta} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                    <IconInstagram />
                    <span className="font-kyobo text-sm">인스타그램</span>
                </button>
                <button onClick={handleDownloadImage} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                    <IconSave />
                    <span className="font-kyobo text-sm">사진 저장</span>
                </button>
            </div>

            <div className="flex items-center gap-8">
                <button onClick={onRetry} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                    다시 만들기
                </button>
                <button onClick={onGoHome} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                    첫 화면으로
                </button>
            </div>
        </div>
    );
}