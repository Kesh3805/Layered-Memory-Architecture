/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sidebar: {
          bg: '#0a0a0a',
          hover: '#1a1a1a',
          active: '#242424',
          text: '#e4e4e7',
          muted: '#71717a',
          border: '#27272a',
        },
        chat: {
          bg: '#09090b',
          surface: '#18181b',
          msg: '#1c1c1f',
        },
        accent: {
          DEFAULT: '#10b981',
          hover: '#059669',
          muted: 'rgba(16, 185, 129, 0.12)',
          glow: 'rgba(16, 185, 129, 0.25)',
        },
        input: {
          bg: '#18181b',
          border: '#3f3f46',
          focus: '#52525b',
        },
        danger: {
          DEFAULT: '#ef4444',
          muted: 'rgba(239, 68, 68, 0.12)',
        },
        surface: {
          0: '#09090b',
          1: '#18181b',
          2: '#27272a',
          3: '#3f3f46',
        },
      },
      fontFamily: {
        sans: [
          'Inter', 'SF Pro Display', '-apple-system', 'BlinkMacSystemFont',
          'Segoe UI', 'Roboto', 'Helvetica Neue', 'sans-serif',
        ],
        mono: [
          'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code',
          'Consolas', 'monospace',
        ],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
      boxShadow: {
        'glow-sm': '0 0 12px -2px rgba(16, 185, 129, 0.20)',
        'glow-md': '0 0 24px -4px rgba(16, 185, 129, 0.25)',
        'glow-lg': '0 0 48px -8px rgba(16, 185, 129, 0.30)',
        'elevated': '0 8px 32px -4px rgba(0, 0, 0, 0.5)',
        'elevated-lg': '0 24px 64px -8px rgba(0, 0, 0, 0.6)',
        'inner-glow': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.04)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(ellipse at center, var(--tw-gradient-stops))',
        'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E\")",
      },
      animation: {
        'fade-in': 'fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) both',
        'slide-up': 'slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both',
        'scale-in': 'scaleIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) both',
        'glow-pulse': 'glowPulse 2.5s ease-in-out infinite',
        'shimmer': 'shimmer 2.5s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'breathe': 'breathe 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          from: { opacity: '0', transform: 'scale(0.95)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(16, 185, 129, 0)' },
          '50%': { boxShadow: '0 0 20px 4px rgba(16, 185, 129, 0.15)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '200% 0' },
          '100%': { backgroundPosition: '-200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-6px)' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};
