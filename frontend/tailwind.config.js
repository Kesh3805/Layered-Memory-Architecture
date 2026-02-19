/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        sidebar: {
          bg: '#171717',
          hover: '#2a2a2a',
          active: '#343434',
          text: '#ececec',
          muted: '#8e8e8e',
          border: '#2e2e2e',
        },
        chat: {
          bg: '#212121',
          msg: '#2f2f2f',
        },
        accent: {
          DEFAULT: '#10a37f',
          hover: '#0d8c6d',
        },
        input: {
          bg: '#2f2f2f',
          border: '#424242',
        },
        danger: '#ef4444',
      },
      fontFamily: {
        sans: [
          'Söhne', '-apple-system', 'BlinkMacSystemFont',
          'Segoe UI', 'Roboto', 'Helvetica Neue', 'sans-serif',
        ],
        mono: ['Söhne Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
