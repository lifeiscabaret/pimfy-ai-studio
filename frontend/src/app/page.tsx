'use client';

import { useState, useEffect } from 'react';
import StartPageContent from './StartPageContent';
import MungsaengPage from './mungsaeng/MungsaengPage';
import ProfilePage from './profile/ProfilePage';
import SelectStep from './profile/SelectStep';
import AdoptionCreateStep from './profile/AdoptionCreateStep';   // 핌피바이러스 프로필(검색/선택)
import GeneralCreateStep from './profile/GeneralCreateStep';     // 입양·임보 프로필(수동작성)
import StudioCreateStep from './profile/StudioCreateStep';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ReadyStep from './profile/ReadyStep';

type PageState =
  | 'start'
  | 'mungsaeng'
  | 'profile'
  | 'profileSelect'
  | 'profileAdoptionCreate'   // ✅ 핌피바이러스 프로필
  | 'profileGeneralCreate'    // ✅ 입양·임보 프로필
  | 'studioCreate'
  | 'profileGenerating'
  | 'profileReady';

// (선택) 핌피바이러스 프로필에서 선택한 강아지 보관하고 싶으면 타입 선언
interface DogProfile {
  id: number; name: string; breed: string; age: number;
  story: string; imageUrl: string; shelter: string;
}

export default function Home() {
  const [currentPage, setCurrentPage] = useState<PageState>('start');
  const [selectedDogForProfile, setSelectedDogForProfile] = useState<DogProfile | null>(null);

  const goToHome = () => setCurrentPage('start');

  useEffect(() => {
    if (currentPage === 'profileGenerating') {
      const t = setTimeout(() => setCurrentPage('profileReady'), 3000);
      return () => clearTimeout(t);
    }
  }, [currentPage]);

  // ✅ 선택 화면에서 옵션별로 '다른 상태'로 보냄
  const handleSelectProfileType = (type: 'pimfy' | 'adoption' | 'studio') => {
    console.log('[Select] type =', type);
    if (type === 'pimfy') {
      setCurrentPage('profileAdoptionCreate');   // 핌피바이러스 → AdoptionCreateStep
    } else if (type === 'adoption') {
      setCurrentPage('profileGeneralCreate');    // 입양·임보 → GeneralCreateStep
    } else if (type === 'studio') {
      setCurrentPage('studioCreate');
    }
  };

  // General / Studio 완료 시
  const handleCreateComplete = () => {
    console.log('[Create] general/studio complete');
    setCurrentPage('profileGenerating');
  };

  // 핌피바이러스(검색/선택) 완료 시
  const handleDogSelectionComplete = (dogData: DogProfile) => {
    console.log('[Create] pimfy selected dog =', dogData);
    setSelectedDogForProfile(dogData);
    setCurrentPage('profileGenerating');
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'mungsaeng':
        return <MungsaengPage onBack={() => setCurrentPage('profile')} onGoHome={goToHome} />;

      case 'profileGenerating':
        return <LoadingSpinner mainText="견생 프로필" subText="(사진 생성중)" />;

      case 'profileReady':
        return (
          <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <ReadyStep onRetry={() => setCurrentPage('profileSelect')} onGoHome={goToHome}
            // selectedDog={selectedDogForProfile} // 필요 시 사용
            />
          </div>
        );

      // ✅ 핌피바이러스 프로필 = 검색/선택 화면
      case 'profileAdoptionCreate':
        return (
          <AdoptionCreateStep
            onBack={() => setCurrentPage('profileSelect')}
            onComplete={handleDogSelectionComplete}
          />
        );

      // ✅ 입양·임보 프로필 = 업로드 + 입력 폼 화면
      case 'profileGeneralCreate':
        return (
          <GeneralCreateStep
            onBack={() => setCurrentPage('profileSelect')}
            onComplete={handleCreateComplete}
          />
        );

      case 'studioCreate':
        return <StudioCreateStep onBack={() => setCurrentPage('profileSelect')} onComplete={handleCreateComplete} />;

      case 'profileSelect':
        return <SelectStep onBack={() => setCurrentPage('profile')} onSelect={handleSelectProfileType} />;

      case 'profile':
        return <ProfilePage onBack={goToHome} onNavigate={setCurrentPage} />;

      case 'start':
      default:
        return <StartPageContent onNavigate={setCurrentPage} />;
    }
  };

  return <main>{renderPage()}</main>;
}
