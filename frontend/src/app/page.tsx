'use client';

import { useState, useEffect } from 'react';
import StartPageContent from './StartPageContent';
import MungsaengPage from './mungsaeng/MungsaengPage';
import ProfilePage from './profile/ProfilePage';
import SelectStep from './profile/SelectStep';
import CreateStep from './profile/CreateStep';
import AdoptionCreateStep from './profile/AdoptionCreateStep';
import StudioCreateStep from './profile/StudioCreateStep';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ReadyStep from './profile/ReadyStep';

type PageState = 'start' | 'mungsaeng' | 'profile' | 'profileSelect' | 'profileCreate' | 'profileAdoptionCreate' | 'studioCreate' | 'profileGenerating' | 'profileReady';

export default function Home() {
  const [currentPage, setCurrentPage] = useState<PageState>('start');
  const goToHome = () => setCurrentPage('start');

  useEffect(() => {
    if (currentPage === 'profileGenerating') {
      const timer = setTimeout(() => {
        setCurrentPage('profileReady');
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [currentPage]);

  const handleSelectProfileType = (type: 'pimfy' | 'adoption' | 'studio') => {
    if (type === 'pimfy') {
      setCurrentPage('profileCreate');
    } else if (type === 'adoption') {
      setCurrentPage('profileAdoptionCreate');
    } else if (type === 'studio') {
      setCurrentPage('studioCreate');
    }
  }

  const handleCreateComplete = () => {
    setCurrentPage('profileGenerating');
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'mungsaeng':
        return <MungsaengPage onBack={() => setCurrentPage('profile')} onGoHome={goToHome} />;

      case 'profileGenerating':
        return <LoadingSpinner mainText="견생 프로필" subText="(사진 생성중)" />;

      case 'profileReady':
        return (
          <div className="flex min-h-screen items-center justify-center bg-mint p-4">
            <ReadyStep onRetry={() => setCurrentPage('profileSelect')} onGoHome={goToHome} />
          </div>
        );

      case 'profileCreate':
        return <CreateStep onBack={() => setCurrentPage('profileSelect')} onComplete={handleCreateComplete} />;
      case 'profileAdoptionCreate':
        return <AdoptionCreateStep onBack={() => setCurrentPage('profileSelect')} onComplete={handleCreateComplete} />;
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