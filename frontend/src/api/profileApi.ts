// src/api/profileApi.ts

// 📡 백엔드 서버 주소 (배포 시 IP 확인 필수!)
const API_BASE_URL = "http://211.188.58.199:8000";

// 1️⃣ [입양/임보 프로필] 생성 함수 (contact 추가)
export const createAdoptionProfile = async (
    file: File,
    name: string,
    age: string,
    personality: string,
    features: string,
    contact?: string 
) => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("name", name);
    formData.append("age", age);
    formData.append("personality", personality);
    formData.append("features", features);

    // 연락처가 있으면 전송
    if (contact) formData.append("contact", contact);

    const response = await fetch(`${API_BASE_URL}/api/v1/generate-adoption-profile`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error("입양 프로필 생성 실패");
    return await response.json();
};

// 2️⃣ [스튜디오 프로필] 생성 함수 (기존 동일)
export const createStudioProfile = async (file: File, bgColor: string) => {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("bg_color", bgColor);

    const response = await fetch(`${API_BASE_URL}/api/v1/generate-studio-profile`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) throw new Error("스튜디오 프로필 생성 실패");
    return await response.json();
};

// 3️⃣ [핌피바이러스(자동) 프로필] 생성 함수 (contact 추가)
export const createRealProfile = async (dogUid: number, contact?: string) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/generate-real-profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // body에 contact 추가 전송
        body: JSON.stringify({ dog_uid: dogUid, contact: contact }),
    });

    if (!response.ok) throw new Error("자동 프로필 생성 실패");
    return await response.json();
};

// 4️⃣ [공고 검색] 함수 (기존 동일)
export const searchDogs = async (searchTerm: string) => {
    const response = await fetch(`${API_BASE_URL}/api/dogs/search?name=${searchTerm}`);
    if (!response.ok) throw new Error('서버 통신 실패');
    return await response.json();
};