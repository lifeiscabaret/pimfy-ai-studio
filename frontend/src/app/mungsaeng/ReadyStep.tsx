'use client';

import { useState } from 'react';
import Image from 'next/image';
import { SketchPicker, ColorResult } from 'react-color';

interface ReadyStepProps {
    onRetry: () => void;
    onGoHome: () => void;
}

export default function ReadyStep({ onRetry, onGoHome }: ReadyStepProps) {
    const [frameColor, setFrameColor] = useState('#FF7A4D');
    const [showColorPicker, setShowColorPicker] = useState(false);

    const handleColorChange = (color: ColorResult) => {
        setFrameColor(color.hex);
    };

    const handleDownloadImage = () => alert('사진 다운로드 기능 구현 예정');
    const handleDownloadVideo = () => alert('영상 다운로드 기능 구현 예정');

    return (
        <div className="w-full max-w-5xl flex flex-col items-center gap-6">

            {/* 1행: 사진과 영상 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full">
                {/* 왼쪽: 사진 결과물 */}
                <div className="flex flex-col items-center">
                    <div
                        className="w-full max-w-sm rounded-lg p-3 transition-colors"
                        style={{ backgroundColor: frameColor }}
                    >
                        <div className="relative aspect-[3/4] overflow-hidden rounded-sm bg-white">
                            <Image src="/dog-photo.jpg" alt="완성된 멍생네컷" fill className="object-cover" />
                        </div>
                    </div>
                </div>
                {/* 오른쪽: 영상 결과물 */}
                <div className="flex flex-col items-center">
                    <div
                        className="w-full max-w-sm rounded-lg border-4 p-2 transition-colors"
                        style={{ borderColor: frameColor }}
                    >
                        <div className="aspect-[3/4] bg-gray-100 flex items-center justify-center">
                            <span className="font-bungee text-3xl text-ticket">VIDEO</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* 2행: 색상 변경 컨트롤러 */}
            <div className="relative flex items-center justify-center gap-4">
                <button
                    onClick={() => setShowColorPicker(!showColorPicker)}
                    className="w-10 h-10 rounded-full border-2 border-gray-300"
                    style={{ backgroundColor: frameColor }}
                    aria-label="색상 변경"
                />
                {showColorPicker && (
                    <div className="absolute bottom-full mb-2 z-10">
                        <SketchPicker color={frameColor} onChangeComplete={handleColorChange} />
                    </div>
                )}
                <span className="font-kyobo text-lg text-gray-700">색상 변경</span>
            </div>

            {/* 3행: 모든 버튼들을 하나의 그룹으로 묶어서 관리.*/}
            <div className="w-full flex flex-col items-center gap-8 mt-4">
                {/* 다운로드 버튼들 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-3xl">
                    <button onClick={handleDownloadImage} className="font-kyobo text-xl text-center underline hover:text-ticket transition-colors">사진 다운로드</button>
                    <button onClick={handleDownloadVideo} className="font-kyobo text-xl text-center underline hover:text-ticket transition-colors">영상 다운로드</button>
                </div>
                {/* 다시 찍기 / 첫 화면으로 버튼들 */}
                <div className="flex items-center gap-8">
                    <button onClick={onRetry} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                        다시 찍으러 가기
                    </button>
                    <button onClick={onGoHome} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                        첫 화면으로
                    </button>
                </div>
            </div>

        </div>
    );
}