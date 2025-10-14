'use client';

export type FrameSize = '2x2' | '1x3' | '1x2';

interface SizeStepProps {
    onSizeSelect: (size: FrameSize) => void;
    selectedSize: FrameSize | null;
}

export default function SizeStep({ onSizeSelect, selectedSize }: SizeStepProps) {
    // 각 규격 버튼의 공통 스타일
    const buttonStyle = "p-2 border-4 rounded-lg transition-colors";
    const selectedStyle = "border-ticket";
    const notSelectedStyle = "border-gray-300 hover:border-gray-400";

    return (
        <div className="w-full flex flex-col items-center">
            {/* 상단 텍스트 */}
            <p className="font-kyobo text-3xl text-gray-700 mb-12">
                🐾 원하는 규격을 선택해주세요. 🐾
            </p>

            {/* 규격 선택 영역 */}
            <div className="flex items-center justify-center gap-8 md:gap-12">

                {/* 옵션 1: 2x2 */}
                <button
                    onClick={() => onSizeSelect('2x2')}
                    className={`${buttonStyle} ${selectedSize === '2x2' ? selectedStyle : notSelectedStyle}`}
                >
                    <div className="w-40 h-40 grid grid-cols-2 grid-rows-2 gap-1 bg-gray-200">
                        <div className="bg-white" />
                        <div className="bg-white" />
                        <div className="bg-white" />
                        <div className="bg-white" />
                    </div>
                </button>

                {/* 옵션 2: 1x3 */}
                <button
                    onClick={() => onSizeSelect('1x3')}
                    className={`${buttonStyle} ${selectedSize === '1x3' ? selectedStyle : notSelectedStyle}`}
                >
                    <div className="w-48 h-16 flex gap-1 bg-gray-200">
                        <div className="bg-white flex-1" />
                        <div className="bg-white flex-1" />
                        <div className="bg-white flex-1" />
                    </div>
                </button>

                {/* 옵션 3: 1x2 */}
                <button
                    onClick={() => onSizeSelect('1x2')}
                    className={`${buttonStyle} ${selectedSize === '1x2' ? selectedStyle : notSelectedStyle}`}
                >
                    <div className="w-24 h-40 flex flex-col gap-1 bg-gray-200">
                        <div className="bg-white flex-1" />
                        <div className="bg-white flex-1" />
                    </div>
                </button>

            </div>
        </div>
    );
}