'use client';

import { useState } from 'react';

// 같은 폴더에 있는 파일들 임포트
import ProfilePage from '@/app/profile/ProfilePage';
import SelectStep, { ProfileType } from '@/app/profile/SelectStep';
import AdoptionCreateStep from '@/app/profile/AdoptionCreateStep';
import GeneralCreateStep from '@/app/profile/GeneralCreateStep';
import StudioCreateStep from '@/app/profile/StudioCreateStep';
import ReadyStep from '@/app/profile/ReadyStep';
import StartPageContent from '@/app/StartPageContent'; // ⭐️ [추가] 초기 화면 임포트

// 화면 단계 정의
type ViewState = 'start' | 'main' | 'mungsaeng' | 'select' | 'pimfy' | 'adoption' | 'studio' | 'ready';

export default function Page() {
  // ⭐️ [수정] 초기 상태는 'start' (대문 화면)
  const [view, setView] = useState<ViewState>('start');
  const [resultData, setResultData] = useState<any>(null);

  // 메인 메뉴 네비게이션
  const handleMainNavigate = (page: 'start' | 'mungsaeng' | 'profile' | 'profileSelect') => {
    if (page === 'mungsaeng') setView('mungsaeng');
    if (page === 'profileSelect') setView('select');
  };

  // 프로필 타입 선택
  const handleProfileSelect = (type: ProfileType) => {
    setView(type);
  };

  // 생성 완료 핸들러
  const handleComplete = (data: any) => {
    console.log("생성 완료 데이터:", data);
    setResultData(data);
    setView('ready');
  };

  return (
    <main className="min-h-screen bg-mint">

      {/* 0. ⭐️ 초기 대문 화면 (ENTER 누르면 main으로 이동) */}
      {view === 'start' && (
        <StartPageContent onStart={() => setView('main')} />
      )}

      {/* 1. 메인 메뉴 */}
      {view === 'main' && (
        <ProfilePage
          onBack={() => setView('start')} // 뒤로가기 하면 다시 대문으로
          onNavigate={handleMainNavigate}
        />
      )}

      {/* 2. 멍생네컷 */}
      {view === 'mungsaeng' && (
        <div className="flex h-screen items-center justify-center">
          <div className="text-center">
            <h2 className="font-kyobo text-2xl mb-4">멍생네컷 기능은 준비 중입니다! 📸</h2>
            <button onClick={() => setView('main')} className="bg-white px-4 py-2 rounded shadow">돌아가기</button>
          </div>
        </div>
      )}

      {/* 3. 프로필 타입 선택 */}
      {view === 'select' && (
        <SelectStep
          onSelect={handleProfileSelect}
          onBack={() => setView('main')}
        />
      )}

      {/* 4. 핌피바이러스(공고) 프로필 */}
      {view === 'pimfy' && (
        <AdoptionCreateStep
          onComplete={handleComplete}
          onBack={() => setView('select')}
        />
      )}

      {/* 5. 입양(수동) 프로필 */}
      {view === 'adoption' && (
        <GeneralCreateStep
          onComplete={handleComplete}
          onBack={() => setView('select')}
        />
      )}

      {/* 6. 스튜디오 프로필 */}
      {view === 'studio' && (
        <StudioCreateStep
          onComplete={handleComplete}
          onBack={() => setView('select')}
        />
      )}

      {/* 7. 결과 화면 */}
      {view === 'ready' && (
        <ReadyStep
          profileData={resultData}
          onRetry={() => setView('select')}
          onGoHome={() => setView('main')}
        />
      )}
    </main>
  );
}