'use client';

import { useState } from "react";

interface CreateStepProps {
    onComplete: () => void;
    onBack: () => void;
}

// 백엔드 연동 전 사용할 가짜 검색 결과 데이터
const mockSearchResults = [
    { id: 1, name: '럭키', age: 1, gender: '남아' },
    { id: 2, name: '럭키', age: 5, gender: '여아' },
    { id: 3, name: '럭키', age: 2, gender: '남아' },
    { id: 4, name: '럭키', age: 10, unit: '개월', gender: '여아' },
];

export default function CreateStep({ onComplete, onBack }: CreateStepProps) {
    const [selectedDogId, setSelectedDogId] = useState<number | null>(null);
    const selectedDog = mockSearchResults.find(dog => dog.id === selectedDogId);

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl rounded-2xl bg-white p-8 shadow-lg flex flex-col items-center">

                {/* 상단 타이틀 */}
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-6">
                    <span className="text-brand-pink">♡</span> 핌피바이러스 프로필 <span className="text-brand-pink">♡</span>
                </h1>

                {/* 선택된 강아지 정보 표시 (상단 흰 박스) */}
                <div className="w-full h-32 rounded-2xl bg-gray-50 border-2 border-dashed flex items-center justify-center mb-6">
                    <p className="font-kyobo text-xl text-gray-500">
                        {selectedDog ? `${selectedDog.name} (${selectedDog.age}${selectedDog.unit || '살'} ${selectedDog.gender})` : '아래 목록에서 프로필을 만들 아이를 선택해주세요!'}
                    </p>
                </div>

                {/* 검색 바 */}
                <div className="w-full max-w-lg rounded-full bg-cream p-2 flex items-center gap-2 mb-4">
                    <button className="font-kyobo bg-white rounded-full px-6 py-2 shadow">공고 검색</button>
                    {/* 시안처럼 '럭키'를 예시 검색어로 표시. */}
                    <input
                        type="text"
                        defaultValue="럭키"
                        className="font-kyobo flex-1 bg-transparent text-gray-700 focus:outline-none px-4"
                    />
                </div>

                {/* 검색 결과 목록 */}
                <div className="w-full max-w-lg rounded-lg border-2 border-gray-200 overflow-hidden">
                    {mockSearchResults.map((dog) => (
                        <div
                            key={dog.id}
                            onClick={() => setSelectedDogId(dog.id)}
                            className={`p-4 border-b-2 cursor-pointer transition-colors ${selectedDogId === dog.id ? 'bg-orange-100' : 'hover:bg-gray-50'}`}
                        >
                            <p className="font-kyobo text-lg">
                                {dog.name} {dog.age}{dog.unit || '살'} {dog.gender}
                            </p>
                        </div>
                    ))}
                </div>

                {/* 하단 버튼들 */}
                <div className="w-full max-w-2xl mt-10 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline transition-colors">
                        ← 이전
                    </button>
                    <button
                        onClick={onComplete}
                        disabled={!selectedDog} 
                        className="font-kyobo text-2xl text-gray-800 hover:text-black disabled:text-gray-400 transition-colors"
                    >
                        준비완료! →
                    </button>
                </div>
            </div>
        </div>
    );
}