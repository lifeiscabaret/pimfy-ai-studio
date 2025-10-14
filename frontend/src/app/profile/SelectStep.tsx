'use client';

export type ProfileType = 'pimfy' | 'adoption' | 'studio';

interface SelectStepProps {
    onSelect: (type: ProfileType) => void;
    onBack: () => void;
}

const options = [
    { type: 'pimfy', text: '핌피바이러스 프로필', color: 'bg-ticket', textColor: 'text-white', underlineColor: 'bg-white' },
    { type: 'adoption', text: '입양•임보 프로필', color: 'bg-profile-pink', textColor: 'text-gray-800', underlineColor: 'bg-gray-800' },
    { type: 'studio', text: '스튜디오 프로필', color: 'bg-profile-yellow', textColor: 'text-gray-800', underlineColor: 'bg-gray-800' },
] as const;


export default function SelectStep({ onSelect, onBack }: SelectStepProps) {
    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-10">
                    어떤 프로필을 만들어 볼까요?
                </h1>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {options.map((option) => (
                        <div
                            key={option.type}
                            className={`group flex items-center justify-center aspect-[9/16] rounded-3xl cursor-pointer transition-transform hover:scale-105 ${option.color}`}
                            onClick={() => onSelect(option.type)}
                        >
                            <p className={`font-kyobo text-2xl text-center ${option.textColor}`}>
                                {option.text}
                                <span className={`block max-w-0 group-hover:max-w-full transition-all duration-300 h-0.5 mt-1 mx-auto ${option.underlineColor}`} />
                            </p>
                        </div>
                    ))}
                </div>

                <div className="mt-12 text-center">
                    <button
                        onClick={onBack}
                        className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline transition-colors"
                    >
                        ← 이전 페이지로
                    </button>
                </div>
            </div>
        </div>
    );
}