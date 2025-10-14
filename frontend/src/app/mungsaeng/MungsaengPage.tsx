'use client';

import { useState, useEffect } from 'react';
import UploadStep from './UploadStep';
import SizeStep, { FrameSize } from './SizeStep';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ReadyStep from './ReadyStep';

interface MungsaengPageProps {
    onBack: () => void;
    onGoHome: () => void;
}

export default function MungsaengPage({ onBack, onGoHome }: MungsaengPageProps) {
    const [currentStep, setCurrentStep] = useState('upload');
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [selectedSize, setSelectedSize] = useState<FrameSize | null>(null);

    useEffect(() => {
        if (currentStep === 'generating') {
            const timer = setTimeout(() => {
                setCurrentStep('ready');
            }, 3000);
            return () => clearTimeout(timer);
        }
    }, [currentStep]);

    // "다시 찍으러 가기"를 위한 함수
    const handleRetry = () => {
        setUploadedFile(null);
        setSelectedSize(null);
        setCurrentStep('upload');
    };

    const renderStep = () => {
        switch (currentStep) {
            case 'ready':
                return <ReadyStep onRetry={handleRetry} onGoHome={onGoHome} />;
            case 'generating':
                return <LoadingSpinner mainText="멍생네컷" subText="(사진 생성중)" />;
            case 'size':
                return <SizeStep onSizeSelect={setSelectedSize} selectedSize={selectedSize} />;
            case 'upload':
            default:
                return <UploadStep onFileSelect={setUploadedFile} selectedFile={uploadedFile} />;
        }
    };

    const handleNext = () => {
        if (currentStep === 'upload' && uploadedFile) {
            setCurrentStep('size');
        } else if (currentStep === 'size' && selectedSize) {
            setCurrentStep('generating');
        }
    };

    const handleBack = () => {
        if (currentStep === 'ready' || currentStep === 'size') {
            setCurrentStep('upload');
        } else {
            onBack();
        }
    }

    const nextButtonText = currentStep === 'upload' ? '다음 단계로! →' : '준비완뇨! →';
    const isNextDisabled =
        (currentStep === 'upload' && !uploadedFile) ||
        (currentStep === 'size' && !selectedSize);

    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-mint p-8">
            <div className={currentStep === 'generating' ? "absolute inset-0" : "w-full flex justify-center"}>
                {renderStep()}
            </div>
            {(currentStep !== 'generating' && currentStep !== 'ready') && (
                <div className="absolute bottom-10 flex w-full max-w-2xl justify-between items-center">
                    <button onClick={handleBack} className="font-kyobo text-lg text-gray-600 hover:text-black">
                        ← 뒤로가기
                    </button>
                    <button
                        onClick={handleNext}
                        className="font-kyobo text-xl text-gray-800 hover:text-black disabled:text-gray-400 transition-colors"
                        disabled={isNextDisabled}
                    >
                        {nextButtonText}
                    </button>
                </div>
            )}
        </div>
    );
}