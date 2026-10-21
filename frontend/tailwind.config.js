/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'] },
      colors: { ink: '#16212e', route: '#1d4ed8', signal: '#f59e0b' },
    },
  },
  plugins: [],
};
