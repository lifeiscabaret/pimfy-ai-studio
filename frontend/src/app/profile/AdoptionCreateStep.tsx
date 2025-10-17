'use client';

import { useState, useEffect } from 'react';
// import Image from 'next/image'; // next/image 임포트 오류를 해결하기 위해 이 줄을 제거하고 일반 <img> 태그 사용.

// 백엔드 API에서 받아올 데이터의 타입을 정의하기 위함.
interface DogProfile {
    id: number;
    name: string;
    breed: string;
    age: number;
    story: string;
    imageUrl: string;
    shelter: string;
}

interface AdoptionCreateStepProps {
    // 강아지 선택이 완료되면 선택된 강아지 정보를 부모 컴포넌트로 전달.
    onComplete: (dog: DogProfile) => void;
    onBack: () => void;
}

export default function AdoptionCreateStep({ onComplete, onBack }: AdoptionCreateStepProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState<DogProfile[]>([]);
    const [selectedDog, setSelectedDog] = useState<DogProfile | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // 검색 버튼을 눌렀을 때 API를 호출하는 함수
    const handleSearch = async () => {
        if (!searchTerm.trim()) {
            setSearchResults([]);
            return;
        }
        setIsLoading(true);
        setError(null);
        try {
            // 백엔드 서버(localhost:8000)의 검색 API를 호출.
            const response = await fetch(`http://localhost:8000/api/dogs/search?name=${searchTerm}`);
            if (!response.ok) {
                throw new Error('서버에서 데이터를 가져오는 데 실패했습니다.');
            }
            const data: DogProfile[] = await response.json();
            setSearchResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.');
            setSearchResults([]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl rounded-2xl bg-white p-8 shadow-lg flex flex-col items-center">
                {/* 상단 타이틀 UI는 그대로 유지 */}
                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-6">
                    <span className="text-brand-pink">♡</span> 핌피바이러스 프로필 <span className="text-brand-pink">♡</span>
                </h1>

                {/* 선택된 강아지 정보 표시 UI는 그대로 유지 */}
                <div className="w-full h-32 rounded-2xl bg-gray-50 border-2 border-dashed flex items-center justify-center mb-6 p-4 text-center">
                    <p className="font-kyobo text-xl text-gray-500">
                        {selectedDog ? `${selectedDog.name} (${selectedDog.breed}, ${selectedDog.age}살)` : '프로필을 만들 아이의 이름을 검색해주세요!'}
                    </p>
                </div>

                {/* 검색 바 UI는 그대로 유지, 기능만 연결 */}
                <div className="w-full max-w-lg rounded-full bg-cream p-2 flex items-center gap-2 mb-4">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="이름으로 검색"
                        className="font-kyobo flex-1 bg-transparent text-gray-700 focus:outline-none px-4"
                    />
                    <button onClick={handleSearch} className="font-kyobo bg-white rounded-full px-6 py-2 shadow hover:bg-gray-100 transition-colors">
                        {isLoading ? '검색중...' : '공고 검색'}
                    </button>
                </div>

                {/* 검색 결과 목록 UI는 그대로 유지, 데이터만 API 결과로 표시 */}
                <div className="w-full max-w-lg rounded-lg border-2 border-gray-200 overflow-y-auto max-h-60">
                    {error && <p className="p-4 text-red-500 text-center">{error}</p>}
                    {!error && searchResults.length === 0 && !isLoading && (
                        <p className="p-4 text-gray-400 text-center">검색 결과가 없습니다.</p>
                    )}
                    {searchResults.map((dog) => (
                        <div
                            key={dog.id}
                            onClick={() => setSelectedDog(dog)}
                            className={`p-4 border-b-2 cursor-pointer transition-colors flex items-center gap-4 ${selectedDog?.id === dog.id ? 'bg-orange-100' : 'hover:bg-gray-50'}`}
                        >
                            {/* [오류 수정] Next.js의 Image 컴포넌트 대신 일반 HTML <img> 태그를 사용합니다. */}
                            <img src={dog.imageUrl} alt={dog.name} width={50} height={50} className="rounded-md object-cover" />
                            <div>
                                <p className="font-kyobo text-lg font-bold">{dog.name} ({dog.breed}, {dog.age}살)</p>
                                <p className="font-kyobo text-sm text-gray-600">{dog.shelter}</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* 하단 버튼들 UI는 그대로 유지 */}
                <div className="w-full mt-10 flex justify-between items-center">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-600 hover:text-black hover:underline transition-colors">
                        ← 이전
                    </button>
                    <button
                        onClick={() => selectedDog && onComplete(selectedDog)}
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
