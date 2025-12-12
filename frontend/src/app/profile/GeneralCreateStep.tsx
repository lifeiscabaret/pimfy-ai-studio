'use client';

import { useRef, useState, ChangeEvent, useEffect } from 'react';
import { createAdoptionProfile } from '@/api/profileApi';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface GeneralCreateStepProps {
    onComplete: (data: any) => void;
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
    const [contact, setContact] = useState(''); // ⭐️ 연락처 상태 추가

    const [isLoading, setIsLoading] = useState(false);

    // 미리보기 URL 메모리 해제
    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    const handleUploadClick = () => fileInputRef.current?.click();

    // ⭐️ HEIC 이미지 처리 로직 (동적 import)
    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        setFile(selectedFile);

        // HEIC 파일인지 확인
        if (selectedFile.name.toLowerCase().endsWith('.heic') || selectedFile.type === 'image/heic') {
            try {
                // 필요할 때만 라이브러리 로딩 (SSR 에러 방지)
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
                // 실패 시 원본으로 시도
                setPreviewUrl(URL.createObjectURL(selectedFile));
            }
        } else {
            setPreviewUrl(URL.createObjectURL(selectedFile));
        }
    };

    // 프로필 생성 요청
    const handleGenerate = async () => {
        if (!file || !name || !age || !personality || !features) return;

        setIsLoading(true);
        try {
            // ⭐️ contact 정보 함께 전송
            const result = await createAdoptionProfile(file, name, age, personality, features, contact);
            onComplete(result);
        } catch (error) {
            alert("프로필 생성 중 오류가 발생했습니다.");
            console.error(error);
        } finally {
            setIsLoading(false);
        }
    };

    const isReady = file && name && age && personality && features;

    if (isLoading) {
        return (
            <LoadingSpinner
                mainText={`AI가 ${name}의 프로필을 예쁘게 꾸미고 있어요!`}
                subText="(약 20~30초 정도 걸려요 🐶)"
            />
        );
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-8">
                    <span className="text-profile-pink">♡</span> 입양•임보 프로필 <span className="text-profile-pink">♡</span>
                </h1>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* 왼쪽: 이미지 업로드 영역 */}
                    <div
                        onClick={handleUploadClick}
                        className="w-full aspect-[3/4] rounded-2xl bg-gray-50 border-2 border-dashed flex items-center justify-center cursor-pointer hover:border-ticket overflow-hidden relative"
                    >
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*,.heic" />
                        {previewUrl ? (
                            <img src={previewUrl} alt="업로드 미리보기" className="w-full h-full object-cover" />
                        ) : (
                            <span className="font-bold text-2xl text-ticket tracking-widest">UPLOAD</span>
                        )}
                    </div>

                    {/* 오른쪽: 정보 입력 영역 */}
                    <div className="flex flex-col justify-center gap-4">
                        <input type="text" placeholder="이름" value={name} onChange={(e) => setName(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="나이 (예: 2살)" value={age} onChange={(e) => setAge(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="성격 (예: 활발함, 애교쟁이)" value={personality} onChange={(e) => setPersonality(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />
                        <input type="text" placeholder="특징 (예: 귀가 접힘)" value={features} onChange={(e) => setFeatures(e.target.value)} className="font-kyobo text-xl p-4 bg-cream rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket" />

                        {/* ⭐️ 문의처 입력 칸 추가 */}
                        <input
                            type="text"
                            placeholder="문의처/SNS (선택사항)"
                            value={contact}
                            onChange={(e) => setContact(e.target.value)}
                            className="font-kyobo text-xl p-4 bg-white border-2 border-brand-pink/30 rounded-2xl focus:outline-none focus:ring-2 focus:ring-ticket placeholder:text-gray-400"
                        />
                    </div>
                </div>
                <div className="w-full mt-8 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline">
                        ← 이전
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={!isReady}
                        className="font-kyobo text-2xl text-gray-800 hover:text-black disabled:text-gray-400"
                    >
                        프로필 생성하기! →
                    </button>
                </div>
            </div>
        </div>
    );
}