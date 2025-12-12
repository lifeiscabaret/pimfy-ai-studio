'use client';

import { useRef, useState, ChangeEvent, useEffect } from 'react';
import { createStudioProfile } from '@/api/profileApi';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface StudioCreateStepProps {
    onComplete: (data: any) => void;
    onBack: () => void;
}

export default function StudioCreateStep({ onComplete, onBack }: StudioCreateStepProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const [bgColor, setBgColor] = useState('#FFD1DC');
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    const handleUploadClick = () => fileInputRef.current?.click();

    // ⭐️ [핵심 수정] HEIC 동적 로딩 적용
    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        setFile(selectedFile);

        if (selectedFile.name.toLowerCase().endsWith('.heic') || selectedFile.type === 'image/heic') {
            try {
                const heic2any = (await import('heic2any')).default;

                const convertedBlob = await heic2any({
                    blob: selectedFile,
                    toType: 'image/jpeg',
                });
                const blob = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
                const url = URL.createObjectURL(blob);
                setPreviewUrl(url);
            } catch (err) {
                console.error("HEIC 변환 실패:", err);
                setPreviewUrl(URL.createObjectURL(selectedFile));
            }
        } else {
            setPreviewUrl(URL.createObjectURL(selectedFile));
        }
    };

    const handleGenerate = async () => {
        if (!file) return;
        setIsLoading(true);
        try {
            const result = await createStudioProfile(file, bgColor);
            onComplete(result);
        } catch (error) {
            alert("스튜디오 프로필 생성 실패!");
            console.error(error);
        } finally {
            setIsLoading(false);
        }
    };

    if (isLoading) {
        return (
            <LoadingSpinner
                mainText="스튜디오 촬영 중... 📸"
                subText="(잠시만 기다려주세요!)"
            />
        );
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl rounded-2xl bg-white p-8 shadow-lg flex flex-col items-center">

                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-8">
                    <span className="text-brand-pink">♡</span> 스튜디오 프로필 <span className="text-brand-pink">♡</span>
                </h1>

                {/* 이미지 업로드 */}
                <div
                    onClick={handleUploadClick}
                    className="w-full max-w-sm aspect-square rounded-3xl bg-gray-50 border-2 border-dashed flex items-center justify-center cursor-pointer hover:border-ticket mb-6 overflow-hidden relative"
                >
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*,.heic" />
                    {previewUrl ? (
                        <img src={previewUrl} alt="업로드 미리보기" className="w-full h-full object-cover" />
                    ) : (
                        <span className="font-bold text-2xl text-ticket tracking-widest">UPLOAD</span>
                    )}
                </div>

                {/* 배경색 선택 */}
                <div className="w-full max-w-sm flex items-center justify-between bg-cream p-4 rounded-xl mb-6">
                    <span className="font-kyobo text-xl text-gray-700">배경색 선택</span>
                    <input
                        type="color"
                        value={bgColor}
                        onChange={(e) => setBgColor(e.target.value)}
                        className="w-10 h-10 rounded-full cursor-pointer border-none bg-transparent"
                    />
                </div>

                {/* 하단 버튼 */}
                <div className="w-full mt-4 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline">
                        ← 이전
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={!file}
                        className="font-kyobo text-2xl text-gray-800 hover:text-black disabled:text-gray-400"
                    >
                        촬영하기! →
                    </button>
                </div>

            </div>
        </div>
    );
}