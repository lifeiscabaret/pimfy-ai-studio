'use client';

import { useState, useEffect } from 'react';

export default function Home() {
  const [message, setMessage] = useState('로딩 중...');

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/test')
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message);
      })
      .catch(() => {
        setMessage('백엔드 서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요.');
      });
  }, []);

  return (
    <main style={{ padding: '2rem' }}>
      <h1>Pimfy AI 프로필 스튜디오🐾</h1>
      <p>백엔드로부터 받은 메시지:</p>
      <p style={{ color: 'blue', fontWeight: 'bold' }}>{message}</p>
    </main>
  );
}