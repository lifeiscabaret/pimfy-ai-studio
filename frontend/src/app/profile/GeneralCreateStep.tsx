'use client';

import { useRef, useState, ChangeEvent, useEffect } from 'react';

interface GeneralCreateStepProps {
    onComplete: () => void;
    onBack: () => void;
}

export default function GeneralCreateStep({ onComplete, onBack }: GeneralCreateStepProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    const [name, setName] = useState('');
    const [age, setAge] = useState('');
    const [personality, setPersonality] = useState('');
    const [features, setFeatures] = useState('');

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

    const isReady = file && name && age && personality && features;

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-8">
                    <span className="text-profile-pink">♡</span> 입양•임보 프로필 <span className="text-profile-pink">♡</span>
                </h1>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div
                        onClick={handleUploadClick}
                        className="w-full aspect-[3/4] rounded-2xl bg-gray-50 border-2 border-dashed flex items-center justify-center cursor-pointer hover:border-ticket"
                    >
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                        {previewUrl ? (
                            <div className="relative w-full h-full p-2">
                                {/* <Image src={previewUrl} alt="업로드 미리보기" fill className="object-contain rounded-lg" /> */}
                                <img src={previewUrl} alt="업로드 미리보기" className="w-full h-full object-contain rounded-lg" />
                            </div>
                        ) : (
                            <span className="font-bold text-2xl text-ticket tracking-widest">UPLOAD</span>
                        )}
                    </div>
                    <div className="flex flex-col justify-center gap-4">
                        <input type="text" placeholder="이름" value={name} onChange={(e) => setName(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="나이" value={age} onChange={(e) => setAge(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="성격" value={personality} onChange={(e) => setPersonality(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="특징" value={features} onChange={(e) => setFeatures(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                    </div>
                </div>
                <div className="w-full mt-8 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline">
                        ← 이전
                    </button>
                    <button
                        onClick={onComplete}
                        disabled={!isReady}
                        className="font-kyobo text-2xl text-gray-800 hover:text-black disabled:text-gray-400"
                    >
                        준비완료! →
                    </button>
                </div>
            </div>
        </div>
    );
}

