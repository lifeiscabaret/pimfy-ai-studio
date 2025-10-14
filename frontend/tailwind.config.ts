import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./src/**/*.{ts,tsx}",
        "./app/**/*.{ts,tsx}",
        "./components/**/*.{ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                mint: "#C8F3E0",       // 배경
                ticket: "#FF7A4D",     // 상단 피켓(오렌지)
                curtain: "#FF9A6A",    // 커튼(살구)
                cream: "#FFF1B8",      // 크림(바닥/출력부)
                'brand-pink': '#F9C9D3',
                'curtain-shadow': '#878585', // (커튼 그림자 색상도 유지)
                'profile-pink': '#FFCAD4',
                'profile-yellow': '#FFF5D1',
            },
            fontFamily: {
                bungee: ['var(--font-bungee)'],
                kyobo: ['var(--font-kyobo)'],
            },
            boxShadow: {
                booth: "6px 6px 0 #FFE8C1", // 시안처럼 둔탁한 그림자
            },
            keyframes: {
                marquee: {
                    '0%': { transform: 'translateX(0%)' },
                    '100%': { transform: 'translateX(-100%)' },
                },
                // 까딱까딱 흔들리는 애니메이션
                wiggle: {
                    '0%, 100%': { transform: 'rotate(-5deg)' },
                    '50%': { transform: 'rotate(5deg)' },
                },
            },
            animation: {
                marquee: 'marquee 30s linear infinite',
                wiggle: 'wiggle 1.5s ease-in-out infinite',
            },
        },
    },
    plugins: [],
};

export default config;