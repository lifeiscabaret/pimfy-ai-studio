'use client';

import { useRef, useState, ChangeEvent, useEffect } from 'react';
import Image from 'next/image';

interface StudioCreateStepProps {
    onComplete: () => void;
    onBack: () => void;
}

export default function StudioCreateStep({ onComplete, onBack }: StudioCreateStepProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setPreviewUrl(null);
        }
    }, [file]);

    const handleUploadClick = () => fileInputRef.current?.click();
    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) setFile(selectedFile);
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl rounded-2xl bg-white p-8 shadow-lg flex flex-col items-center">

                {/* 상단 타이틀 */}
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-8">
                    <span className="text-brand-pink">♡</span> 스튜디오 프로필 <span className="text-brand-pink">♡</span>
                </h1>

                {/* 이미지 업로드 */}
                <div
                    onClick={handleUploadClick}
                    className="w-full max-w-sm aspect-square rounded-3xl bg-gray-50 border-2 border-dashed flex items-center justify-center cursor-pointer hover:border-ticket"
                >
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                    {previewUrl ? (
                        <div className="relative w-full h-full p-2">
                            <Image src={previewUrl} alt="업로드 미리보기" fill className="object-contain rounded-lg" />
                        </div>
                    ) : (
                        <span className="font-bold text-2xl text-ticket tracking-widest">UPLOAD</span>
                    )}
                </div>

                {/* 하단 버튼 */}
                <div className="w-full mt-10 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline">
                        ← 이전
                    </button>
                    <button
                        onClick={onComplete}
                        disabled={!file} 
                        className="font-kyobo text-2xl text-gray-800 hover:text-black disabled:text-gray-400"
                    >
                        준비완료! →
                    </button>
                </div>

            </div>
        </div>
    );
}