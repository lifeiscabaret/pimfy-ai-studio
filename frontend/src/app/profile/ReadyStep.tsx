'use client';

import { useState, useEffect } from 'react';

const IconKakao = () => (<svg viewBox="0 0 32 32" className="w-6 h-6"><path fill="currentColor" d="M16 4.64c-6.96 0-12.64 4.48-12.64 10.08 0 3.52 2.32 6.64 5.76 8.48l-.96 3.52.96-.08 3.2-2.24c1.2.32 2.48.56 3.68.56 6.96 0 12.64-4.48 12.64-10.24S22.96 4.64 16 4.64z" /></svg>);
const IconInstagram = () => (<svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line></svg>);
const IconSave = () => (<svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>);

interface ReadyStepProps {
    profileData: any;
    onRetry: () => void;
    onGoHome: () => void;
}

export default function ReadyStep({ profileData, onRetry, onGoHome }: ReadyStepProps) {

    // 카카오 초기화
    useEffect(() => {
        // @ts-ignore
        if (window.Kakao && !window.Kakao.isInitialized()) {
            // @ts-ignore
            window.Kakao.init('592b68bdf6a6bf3da19b7a6d958723b1');
        }
    }, []);

    const handleShareKakao = () => {
        // @ts-ignore
        if (!window.Kakao || !window.Kakao.isInitialized()) {
            return alert("카카오톡 로딩 중... 잠시 후 다시 시도해주세요.");
        }

        const shareImage = profileData?.image_url || '';
        const currentDomain = window.location.origin; // http://localhost:3000

        // [핵심 변경]
        const resultPageUrl = `${currentDomain}/result?img=${encodeURIComponent(shareImage)}`;

        // @ts-ignore
        window.Kakao.Share.sendDefault({
            objectType: 'feed',
            content: {
                title: '🐶 핌피바이러스 AI 프로필 도착!',
                description: '세상에 단 하나뿐인 우리 아이의 프로필을 확인해보세요! ✨',
                imageUrl: shareImage,
                link: {
                    mobileWebUrl: resultPageUrl,
                    webUrl: resultPageUrl,
                },
            },
            buttons: [
                {
                    title: '프로필 보러가기',
                    link: {
                        mobileWebUrl: resultPageUrl,
                        webUrl: resultPageUrl,
                    },
                },
                {
                    title: '나도 만들기',
                    link: {
                        mobileWebUrl: currentDomain,
                        webUrl: currentDomain,
                    },
                },
            ],
        });
    };

    const handleShareInsta = () => {
        alert("사진을 저장한 뒤 인스타그램에 자랑해주세요! 📸");
        window.location.href = "instagram://app";
    };

    const handleDownloadImage = async () => {
        // [수정] base64 대신 서버에서 준 image_url 사용.
        const downloadUrl = profileData?.image_url;

        if (!downloadUrl) {
            return alert("저장할 이미지 주소가 없습니다. 다시 시도해주세요.");
        }

        try {
            // CORS 문제를 해결하기 위해 fetch로 이미지를 가져와 Blob으로 변환.
            const response = await fetch(downloadUrl);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            const link = document.createElement("a");
            link.href = url;
            link.download = `pimfy_profile_${Date.now()}.jpg`;
            document.body.appendChild(link);
            link.click();

            // 사용한 객체 정리
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("다운로드 에러:", error);
            // 다운로드가 실패시, 새 탭에서 이미지 띄워주기.
            window.open(downloadUrl, '_blank');
        }
    };

    const imgSrc = profileData?.image_url ||
        (profileData?.profile_image_base64 ? `data:image/jpeg;base64,${profileData.profile_image_base64}` : null);

    return (
        <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <div className="w-full max-w-2xl flex flex-col items-center bg-white p-8 rounded-2xl shadow-lg">
                <h1 className="font-kyobo text-4xl mb-6">프로필 완성!</h1>

                <div className="w-full max-w-sm rounded-lg mb-12 overflow-hidden shadow-md bg-gray-100 flex items-center justify-center min-h-[400px]">
                    {imgSrc ? (
                        <img src={imgSrc} alt="완성된 프로필" className="w-full h-auto object-contain" />
                    ) : (
                        <p className="font-kyobo text-2xl text-gray-400">이미지가 없어요 ㅠㅠ</p>
                    )}
                </div>


                {/* 버튼들 */}
                <div className="flex items-center gap-6 mb-10">
                    <button onClick={handleShareKakao} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                        <IconKakao />
                        <span className="font-kyobo text-sm">카톡 공유</span>
                    </button>
                    <button onClick={handleShareInsta} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                        <IconInstagram />
                        <span className="font-kyobo text-sm">인스타그램</span>
                    </button>
                    <button onClick={handleDownloadImage} className="flex flex-col items-center gap-2 text-gray-600 hover:text-black transition-colors">
                        <IconSave />
                        <span className="font-kyobo text-sm">사진 저장</span>
                    </button>
                </div>

                <div className="flex items-center gap-8">
                    <button onClick={onRetry} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                        다시 만들기
                    </button>
                    <button onClick={onGoHome} className="font-kyobo text-xl text-gray-700 hover:text-black hover:underline transition-colors">
                        첫 화면으로
                    </button>
                </div>
            </div>
        </div>
    );
}