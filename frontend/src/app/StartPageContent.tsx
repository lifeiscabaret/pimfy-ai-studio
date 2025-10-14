'use client';

import Image from "next/image";
import { CSSProperties } from "react";

type PageState = 'start' | 'mungsaeng' | 'profile' | 'profileSelect';


const DetailedPawSVG = (props: { size?: number }) => {
    const s = props.size ?? 120;
    return (
        <svg width={s} height={s} viewBox="0 0 303 260" fill="none" xmlns="http://www.w3.org/2000/svg">
            <g filter="url(#filter0_d_71_23)">
                <ellipse cx="27.1603" cy="43.7574" rx="27.1603" ry="43.7574" transform="matrix(0.919646 -0.392748 0.319265 0.947665 0 52.0908)" fill="currentColor" />
                <ellipse cx="117.125" cy="50.8236" rx="27.1594" ry="46.8236" fill="currentColor" />
                <ellipse cx="185.023" cy="52.7347" rx="27.1594" ry="46.8236" fill="currentColor" />
                <ellipse cx="27.0529" cy="46.6746" rx="27.0529" ry="46.6746" transform="matrix(0.928877 0.370388 -0.300063 0.953919 252.742 28.6943)" fill="currentColor" />
                <path fillRule="evenodd" clipRule="evenodd" d="M110.843 161.637C113.333 143.492 124.661 107.203 150.055 107.203C175.449 107.203 185.532 143.492 187.399 161.637C203.582 153.472 235.947 145.307 235.947 177.967C235.947 218.793 226.611 256.897 198.603 237.845H90.3039C76.6109 239.659 50.3453 235.123 54.8267 202.462C60.4283 161.637 86.5694 145.307 110.843 161.637Z" fill="currentColor" />
            </g>
            <defs>
                <filter id="filter0_d_71_23" x="0.32373" y="0" width="302.315" height="259.396" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                    <feFlood floodOpacity="0" result="BackgroundImageFix" />
                    <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
                    <feOffset dy="6" />
                    <feGaussianBlur stdDeviation="5" />
                    <feComposite in2="hardAlpha" operator="out" />
                    <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
                    <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_71_23" />
                    <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow_71_23" result="shape" />
                </filter>
            </defs>
        </svg>
    );
};

const BoneSVG = (props: { width?: number }) => {
    const w = props.width ?? 360;
    const h = (w * 140) / 360;
    return (
        <svg width={w} height={h} viewBox="0 0 360 140" fill="currentColor">
            <path d="M40,70 C20,30 70,10 90,38 L270,38 C290,10 340,30 320,70 C340,110 290,130 270,102 L90,102 C70,130 20,110 40,70 Z" fill="currentColor" />
        </svg>
    );
};

const DetailedCurtainSVG = ({ className }: { className?: string }) => (
    <svg className={className} viewBox="0 0 580 407" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <path fillRule="evenodd" clipRule="evenodd" d="M1 385.67V1H579V405L560.709 385.67H522.297C511.323 385.67 507.665 385.67 491.203 376.005C481.657 370.401 478.262 374.545 474.598 379.017C471.942 382.257 469.146 385.67 463.766 385.67H421.696C417.891 385.67 414.526 387.529 411.296 389.313C405.209 392.675 399.6 395.773 392.43 385.67C381.456 370.206 357.677 376.005 321.095 385.67C310.12 385.67 302.804 385.67 300.975 376.005C299.511 368.273 284.513 379.227 277.196 385.67C269.88 385.67 255.613 383.737 257.076 376.005C258.185 370.146 246.522 374.942 235.128 379.628C227.726 382.672 220.437 385.67 216.835 385.67C212.263 385.67 207.69 383.254 203.117 380.837C198.544 378.421 193.972 376.005 189.399 376.005C182.082 376.005 174.156 382.448 171.108 385.67L154.646 376.005L132.696 385.67L107.089 376.005L88.7975 385.67L61.3608 376.005L39.4114 385.67L24.7785 376.005L1 385.67Z" fill="#FF9A6A" />
        <path d="M560.709 385.67L579 405V1H1V385.67L24.7785 376.005L39.4114 385.67L61.3608 376.005L88.7975 385.67L107.089 376.005L132.696 385.67L154.646 376.005L171.108 385.67C174.156 382.448 182.082 376.005 189.399 376.005C193.972 376.005 198.544 378.421 203.117 380.837C207.69 383.254 212.263 385.67 216.835 385.67C220.437 385.67 227.726 382.672 235.128 379.628C246.522 374.942 258.185 370.146 257.076 376.005C255.613 383.737 269.88 385.67 277.196 385.67C284.513 379.226 299.511 368.273 300.975 376.005C302.804 385.67 310.12 385.67 321.095 385.67C357.677 376.005 381.456 370.206 392.43 385.67C399.6 395.773 405.209 392.675 411.296 389.313C414.526 387.529 417.891 385.67 421.696 385.67M560.709 385.67H522.297M560.709 385.67C551.563 385.67 531.077 385.67 522.297 385.67M522.297 385.67C511.323 385.67 507.665 385.67 491.203 376.005C481.657 370.401 478.262 374.545 474.598 379.017C471.942 382.257 469.146 385.67 463.766 385.67M463.766 385.67H421.696M463.766 385.67C450.962 385.67 432.671 385.67 421.696 385.67" stroke="#263238" strokeOpacity="0.01" />
        <path fillRule="evenodd" clipRule="evenodd" d="M71.1377 57.4125C65.6954 97.9159 56.4845 183.129 63.1798 199.954C63.8134 204.409 64.2748 213.56 64.8516 224.999C66.5818 259.316 69.3501 314.223 80.923 324.698C96.3535 338.664 81.6796 55.9131 71.1377 57.4125Z" fill="#878585" fillOpacity="0.54" />
        <path fillRule="evenodd" clipRule="evenodd" d="M196.697 108.275C193.82 148.414 189.493 232.932 195.203 249.897C195.837 254.352 196.491 263.476 197.308 274.881C199.76 309.095 203.682 363.838 212.947 374.641C225.299 389.045 204.776 107.126 196.697 108.275Z" fill="#878585" fillOpacity="0.54" />
        <path fillRule="evenodd" clipRule="evenodd" d="M303.697 54.2751C300.82 94.4136 296.493 178.932 302.203 195.897C302.837 200.352 303.491 209.476 304.308 220.881C306.76 255.095 310.682 309.838 319.947 320.641C332.299 335.045 311.776 53.1259 303.697 54.2751Z" fill="#878585" fillOpacity="0.54" />
        <path fillRule="evenodd" clipRule="evenodd" d="M429.697 108.275C426.82 148.414 422.493 232.932 428.203 249.897C428.837 254.352 429.491 263.476 430.308 274.881C432.76 309.095 436.682 363.838 445.947 374.641C458.299 389.045 437.776 107.126 429.697 108.275Z" fill="#878585" fillOpacity="0.54" />
    </svg>
);


export default function StartPageContent({ onNavigate }: { onNavigate: (page: PageState) => void; }) {
    return (
        <div className="flex flex-col min-h-screen items-center justify-center p-4 bg-mint gap-8">
            <div className="relative w-full max-w-5xl" style={{ height: '120px' }}>
                <div className="absolute top-0 left-1/2 -translate-x-1/2">
                    <div className="bg-ticket rounded-[13px] px-10 py-4 shadow-booth">
                        <h1 className="font-bungee text-[96px] leading-none tracking-[.06em] text-cream text-shadow-cream whitespace-nowrap">
                            PIMFY PHOTO
                        </h1>
                    </div>
                </div>
            </div>
            <div className="relative w-full max-w-5xl aspect-[16/9] rounded-2xl overflow-hidden bg-white shadow-md">
                <div className="grid h-full grid-cols-[300px_1fr]">
                    <div className="flex flex-col items-center bg-ticket p-5">
                        <div className="w-full max-w-[280px] rounded-[14px] ring-8 ring-orange-400 ring-offset-[10px] ring-offset-ticket">
                            <div className="rounded-[10px] bg-cream p-4">
                                <div className="mb-3 text-center text-brand-pink">
                                    <span className="font-kyobo text-[24px] font-bold">♡ 멍생네컷 🐾</span>
                                </div>
                                <div className="relative aspect-[3/4] overflow-hidden rounded-md bg-white">
                                    <Image src={"/dog-photo.jpg"} alt="sample-1" fill className="object-cover" priority />
                                </div>
                            </div>
                        </div>
                        <div className="mt-8 w-full max-w-[280px] flex items-end gap-3 justify-center">
                            <div className="flex-1 h-10 rounded-md bg-cream border-2 border-black/80 flex items-center">
                                <div className="w-full h-[2px] bg-black/80" />
                            </div>
                            <div className="w-12 h-16 rounded-md bg-cream border-2 border-black/80" />
                        </div>
                    </div>
                    <div className="relative bg-white">
                        <div className="absolute top-0 right-0 bottom-10 w-4/5">
                            <DetailedCurtainSVG className="w-full h-full" />
                        </div>
                        <div className="absolute bottom-0 left-0 right-0">
                            <div className="h-10 bg-cream rounded-t-[40%_50%]" />
                        </div>
                    </div>
                </div>
                <div className="absolute inset-0 z-20 flex items-center justify-center">
                    <button onClick={() => onNavigate('profile')} className="relative group focus:outline-none flex flex-col items-center" aria-label="Enter">
                        <div className="mb-0 drop-shadow text-brand-pink transition-transform duration-300 group-hover:-rotate-12">
                            <DetailedPawSVG size={120} />
                        </div>
                        <div className="relative transition-transform duration-300 ease-in-out group-hover:-rotate-12 group-hover:scale-105 text-brand-pink">
                            <BoneSVG width={270} />
                            <span className="font-bungee absolute inset-0 flex items-center justify-center text-[28px] text-cream text-emboss" style={{ transform: "translateY(-4px)" }}>
                                ENTER
                            </span>
                        </div>
                    </button>
                </div>
            </div>
            <p className="font-kyobo text-center text-gray-800 text-[32px]">
                단 한 장의 사진으로 시작하는 우리 아이들의 프로필!
            </p>
        </div>
    );
}