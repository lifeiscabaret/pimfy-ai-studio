'use client';

import { useState } from 'react';
import { createRealProfile, searchDogs } from '@/api/profileApi';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface DogProfile { id: number; name: string; breed: string; age: number; story: string; imageUrl: string; shelter: string; }
interface AdoptionCreateStepProps { onComplete: (data: any) => void; onBack: () => void; }

export default function AdoptionCreateStep({ onComplete, onBack }: AdoptionCreateStepProps) {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState<DogProfile[]>([]);
    const [selectedDog, setSelectedDog] = useState<DogProfile | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [imageErrorIds, setImageErrorIds] = useState<number[]>([]);
    const [contact, setContact] = useState(''); // 연락처 상태

    const handleSearch = async () => {
        if (!searchTerm.trim()) { setSearchResults([]); return; }
        setIsLoading(true); setError(null); setImageErrorIds([]);
        try {
            const data: DogProfile[] = await searchDogs(searchTerm);
            setSearchResults(data);
        } catch (err) { setError(err instanceof Error ? err.message : '오류'); setSearchResults([]); }
        finally { setIsLoading(false); }
    };

    const handleGenerate = async () => {
        if (!selectedDog) return;
        setIsGenerating(true);
        try {
            // contact 추가 전송
            const result = await createRealProfile(selectedDog.id, contact);
            onComplete(result);
        } catch (err) {
            console.error(err);
            alert("생성 실패");
        } finally {
            setIsGenerating(false);
        }
    };

    const handleImageError = (id: number) => { setImageErrorIds(prev => [...prev, id]); };

    if (isGenerating) {
        return <LoadingSpinner mainText={`${selectedDog?.name}의 정보를 분석 중이에요!`} subText="조금만 기다려주세요 🐾" />;
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl rounded-2xl bg-white p-8 shadow-lg flex flex-col items-center">

                <h1 className="font-kyobo text-3xl text-center text-gray-800 mb-8">
                    <span className="text-brand-pink">♡</span> 핌피바이러스 프로필 <span className="text-brand-pink">♡</span>
                </h1>

                {/* 1. 선택된 강아지 정보 */}
                <div className="w-full min-h-[220px] flex items-center justify-center mb-6 relative">
                    <div className="absolute inset-0 bg-cream rounded-3xl opacity-30 transform rotate-1"></div>

                    {selectedDog ? (
                        <div className="z-10 flex flex-col items-center gap-4 animate-fadeIn">
                            {/* 이미지 */}
                            <div className="relative shrink-0">
                                {imageErrorIds.includes(selectedDog.id) ? (
                                    <div className="w-48 h-48 rounded-full bg-gray-200 border-4 border-white shadow-xl flex items-center justify-center text-gray-400 font-kyobo text-xl">
                                        이미지 없음
                                    </div>
                                ) : (
                                    <img
                                        src={selectedDog.imageUrl}
                                        alt={selectedDog.name}
                                        className="w-48 h-48 rounded-full object-cover border-4 border-white shadow-xl ring-4 ring-brand-pink/20"
                                        onError={() => handleImageError(selectedDog.id)}
                                    />
                                )}
                                <div className="absolute bottom-0 right-0 bg-white rounded-full p-2 shadow-md text-2xl">🐾</div>
                            </div>

                            {/* 텍스트 */}
                            <div className="text-center space-y-1">
                                <p className="font-kyobo text-3xl text-gray-800 font-bold">
                                    {selectedDog.name} <span className="text-xl text-gray-500 font-normal">({selectedDog.age}살)</span>
                                </p>
                                <p className="font-kyobo text-lg text-brand-pink">{selectedDog.breed}</p>
                                <p className="font-kyobo text-sm text-gray-400">{selectedDog.shelter}</p>
                            </div>
                        </div>
                    ) : (
                        <div className="z-10 text-center text-gray-400 font-kyobo space-y-3 p-8 border-2 border-dashed border-gray-300 rounded-3xl w-full h-full flex flex-col items-center justify-center">
                            <div className="text-5xl opacity-50">🐕</div>
                            <p className="text-xl">아래에서 아이 이름을 검색해주세요!</p>
                        </div>
                    )}
                </div>

                {/* 2. 검색 바 */}
                <div className="w-full max-w-lg rounded-full bg-cream p-2 flex items-center gap-2 mb-6 shadow-sm border border-brand-pink/10">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="이름으로 검색 (예: 밤이)"
                        className="font-kyobo flex-1 bg-transparent text-gray-700 focus:outline-none px-6 text-lg placeholder:text-gray-400"
                    />
                    <button
                        onClick={handleSearch}
                        className="font-kyobo bg-white rounded-full px-8 py-3 shadow-md hover:bg-brand-pink hover:text-white transition-all text-gray-700 font-bold"
                    >
                        {isLoading ? '...' : '공고 검색'}
                    </button>
                </div>

                {/* 3. 검색 결과 목록 */}
                <div className="w-full max-w-lg rounded-xl border border-gray-200 overflow-y-auto max-h-60 bg-white shadow-inner custom-scrollbar mb-6">
                    {!isLoading && searchResults.length === 0 && searchTerm && (
                        <p className="text-center py-8 text-gray-400 font-kyobo">검색 결과가 없습니다.</p>
                    )}
                    {searchResults.map((dog) => (
                        <div
                            key={dog.id}
                            onClick={() => setSelectedDog(dog)}
                            className={`p-4 border-b last:border-b-0 cursor-pointer transition-all flex items-center gap-4 hover:bg-orange-50 ${selectedDog?.id === dog.id ? 'bg-orange-100' : ''}`}
                        >
                            {/* 썸네일 */}
                            {imageErrorIds.includes(dog.id) ? (
                                <div className="w-14 h-14 rounded-xl bg-gray-200 flex-shrink-0" />
                            ) : (
                                <img
                                    src={dog.imageUrl}
                                    alt={dog.name}
                                    className="w-14 h-14 rounded-xl object-cover flex-shrink-0 shadow-sm border border-gray-100"
                                    onError={() => handleImageError(dog.id)}
                                />
                            )}
                            <div>
                                <p className="font-kyobo text-lg font-bold text-gray-800">{dog.name} <span className="text-sm font-normal text-gray-500">({dog.age}살)</span></p>
                                <p className="font-kyobo text-sm text-gray-500 line-clamp-1">{dog.shelter}</p>
                            </div>
                        </div>
                    ))}
                </div>

                {/* 4. 문의처 입력란 (선택) */}
                {selectedDog && (
                    <div className="w-full max-w-lg animate-fadeIn mb-4">
                        <div className="bg-gray-50 p-4 rounded-xl border border-brand-pink/20 flex flex-col items-center gap-2">
                            <label className="font-kyobo text-brand-pink font-bold text-lg">💌 프로필에 넣을 연락처 (선택)</label>
                            <input
                                type="text"
                                placeholder="예: @instagram_id 또는 010-1234-5678"
                                value={contact}
                                onChange={(e) => setContact(e.target.value)}
                                className="font-kyobo w-full px-4 py-3 rounded-lg border border-gray-300 text-center focus:outline-none focus:ring-2 focus:ring-brand-pink placeholder:text-gray-400 bg-white"
                            />
                        </div>
                    </div>
                )}

                {/* 5. 하단 버튼 */}
                <div className="w-full flex justify-between items-center px-4">
                    <button onClick={onBack} className="font-kyobo text-lg text-gray-500 hover:text-black hover:underline transition-colors">
                        ← 이전
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={!selectedDog}
                        className="font-kyobo text-2xl text-white bg-brand-pink px-8 py-3 rounded-full shadow-lg hover:bg-opacity-90 disabled:bg-gray-300 disabled:shadow-none transition-all transform hover:scale-105 active:scale-95"
                    >
                        준비완료! →
                    </button>
                </div>
            </div>
        </div>
    );
}