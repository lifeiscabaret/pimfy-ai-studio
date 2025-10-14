'use client';

import Image from 'next/image';

type PageState = 'start' | 'mungsaeng' | 'profile' | 'profileSelect';

interface ProfilePageProps {
    onBack: () => void;
    onNavigate: (page: PageState) => void;
}

export default function ProfilePage({ onBack, onNavigate }: ProfilePageProps) {
    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                    {/* ✅ '멍생네컷' 클릭 시 onNavigate('mungsaeng') 호출 */}
                    <button
                        onClick={() => onNavigate('mungsaeng')}
                        className="flex flex-col items-center p-4 rounded-lg hover:bg-gray-100 transition-colors focus:outline-none"
                    >
                        <div className="w-full max-w-xs rounded-lg bg-ticket p-2">
                            <div className="relative aspect-[3/4] overflow-hidden rounded-sm bg-white">
                                <Image src="/dog-photo.jpg" alt="멍생네컷 예시" fill className="object-cover" />
                            </div>
                        </div>
                        <p className="font-kyobo text-2xl mt-4 text-gray-800">
                            <span className="text-brand-pink">♡</span> 멍생네컷 찍으러가기 <span className="text-brand-pink">♡</span>
                        </p>
                    </button>

                    {/* ✅ '프로필 제작' 클릭 시 onNavigate('profileSelect') 호출 */}
                    <button
                        onClick={() => onNavigate('profileSelect')}
                        className="flex flex-col items-center p-4 rounded-lg hover:bg-gray-100 transition-colors focus:outline-none"
                    >
                        <div className="w-full max-w-xs rounded-lg bg-cream p-2">
                            <div className="relative aspect-[3/4] overflow-hidden rounded-md bg-white">
                                <Image src="/profile-sample.jpg" alt="Profile sample" fill className="object-cover" />
                            </div>
                        </div>
                        <p className="font-kyobo text-2xl mt-4 text-gray-800">
                            <span className="text-brand-pink">♡</span> 프로필 제작하러가기 <span className="text-brand-pink">♡</span>
                        </p>
                    </button>

                </div>

                <div className="mt-8 text-center">
                    <button
                        onClick={onBack}
                        className="font-kyobo text-lg text-gray-600 hover:text-black transition-colors"
                    >
                        ← 이전으로 돌아가기
                    </button>
                </div>

            </div>
        </div>
    );
}