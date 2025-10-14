'use client';

import { useRef, useState, ChangeEvent, useEffect } from 'react';
import Image from 'next/image';

interface UploadStepProps {
    onFileSelect: (file: File) => void;
    selectedFile: File | null;
}

export default function UploadStep({ onFileSelect, selectedFile }: UploadStepProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        if (selectedFile) {
            const url = URL.createObjectURL(selectedFile);
            setPreviewUrl(url);

            return () => URL.revokeObjectURL(url);
        } else {
            setPreviewUrl(null);
        }
    }, [selectedFile]);


    // 숨겨진 파일 입력(input)을 클릭하는 함수
    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    // 파일이 실제로 선택되었을 때 호출되는 함수
    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            onFileSelect(file);
        }
    };

    return (
        <div className="w-full flex flex-col items-center">
            {/* 상단 텍스트 */}
            <p className="font-kyobo text-3xl text-gray-700 mb-6">
                🐾 사진을 업로드해주세요. 🐾
            </p>

            {/* 파일 업로드 영역 (클릭 가능) */}
            <div
                onClick={handleUploadClick}
                className="w-full max-w-md aspect-square rounded-2xl bg-white border-2 border-dashed border-gray-300 flex flex-col items-center justify-center cursor-pointer hover:border-ticket hover:bg-orange-50 transition-colors"
            >
                {/* 숨겨진 실제 파일 input */}
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*" // 이미지 파일만 선택 가능하도록 설정
                />

                {previewUrl ? (
                    <div className="relative w-full h-full p-2">
                        <Image
                            src={previewUrl}
                            alt="업로드된 이미지 미리보기"
                            fill
                            className="object-contain rounded-lg"
                        />
                    </div>
                ) : (
                    <span className="font-bold text-2xl text-ticket tracking-widest">
                        UPLOAD
                    </span>
                )}
            </div>
        </div>
    );
}