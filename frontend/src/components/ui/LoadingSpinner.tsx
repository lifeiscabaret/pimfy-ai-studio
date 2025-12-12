'use client';

interface LoadingSpinnerProps {
    mainText: string;
    subText: string;
}

const PawSVG = (props: { className?: string }) => (
    <svg className={props.className} viewBox="0 0 303 260" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g filter="url(#filter0_d_71_23)">
            <ellipse cx="27.16" cy="43.75" rx="27.16" ry="43.75" transform="matrix(0.9196 -0.3927 0.3192 0.9476 0 52.09)" fill="currentColor" />
            <ellipse cx="117.12" cy="50.82" rx="27.15" ry="46.82" fill="currentColor" />
            <ellipse cx="185.02" cy="52.73" rx="27.15" ry="46.82" fill="currentColor" />
            <ellipse cx="27.05" cy="46.67" rx="27.05" ry="46.67" transform="matrix(0.9288 0.3703 -0.3000 0.9539 252.74 28.69)" fill="currentColor" />
            <path fillRule="evenodd" clipRule="evenodd" d="M110.84 161.63C113.33 143.49 124.66 107.2 150.05 107.2C175.44 107.2 185.53 143.49 187.39 161.63C203.58 153.47 235.94 145.3 235.94 177.96C235.94 218.79 226.61 256.89 198.6 237.84H90.3C76.61 239.65 50.34 235.12 54.82 202.46C60.42 161.63 86.56 145.3 110.84 161.63Z" fill="currentColor" />
        </g>
        <defs>
            <filter id="filter0_d_71_23" x="0.32" y="0" width="302.31" height="259.39" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                <feFlood floodOpacity="0" result="BackgroundImageFix" />
                <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
                <feOffset dy="6" /><feGaussianBlur stdDeviation="5" /><feComposite in2="hardAlpha" operator="out" />
                <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" /><feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_71_23" />
                <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow_71_23" result="shape" />
            </filter>
        </defs>
    </svg>
);

const PawRow = () => {
    const paws = Array(6).fill(0);
    return (
        <div className="relative w-full overflow-hidden">
            <div className="flex animate-marquee">
                <div className="flex-shrink-0 flex items-center justify-around min-w-full gap-12 md:gap-32 px-4">
                    {paws.map((_, i) => (
                        <PawSVG
                            key={i}
                            className="animate-wiggle text-brand-pink w-10 h-10 md:w-28 md:h-28 opacity-60"
                        />
                    ))}
                </div>
                <div className="flex-shrink-0 flex items-center justify-around min-w-full gap-12 md:gap-32 px-4">
                    {paws.map((_, i) => (
                        <PawSVG
                            key={i}
                            className="animate-wiggle text-brand-pink w-10 h-10 md:w-28 md:h-28 opacity-60"
                        />
                    ))}
                </div>
            </div>
        </div>
    );
};


export default function LoadingSpinner({ mainText, subText }: LoadingSpinnerProps) {
    return (
        <div className="w-screen h-screen bg-white flex flex-col items-center fixed inset-0 z-50">

            <div className="pt-8 md:pt-16 w-full">
                <PawRow />
            </div>

            <div className="flex-1 flex flex-col items-center justify-center text-center px-6">
                <h1 className="font-kyobo text-3xl md:text-7xl mb-4 md:mb-8 tracking-widest break-keep whitespace-pre-wrap leading-tight">
                    {mainText}
                </h1>

                <p className="font-kyobo text-xl md:text-4xl text-gray-500 break-keep">
                    {subText}
                </p>
            </div>

            <div className="pb-8 md:pb-16 w-full">
                <PawRow />
            </div>

        </div>
    );
}