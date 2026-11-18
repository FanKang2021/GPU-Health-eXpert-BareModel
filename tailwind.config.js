/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './gpu-node-dashboard.tsx',
    "*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        // 科技感主题色彩
        border: "var(--border)",
        input: "var(--input)",
        ring: "var(--ring)",
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: {
          DEFAULT: "var(--primary)",
          foreground: "var(--primary-foreground)",
        },
        secondary: {
          DEFAULT: "var(--secondary)",
          foreground: "var(--secondary-foreground)",
        },
        destructive: {
          DEFAULT: "var(--destructive)",
          foreground: "var(--destructive-foreground)",
        },
        muted: {
          DEFAULT: "var(--muted)",
          foreground: "var(--muted-foreground)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          foreground: "var(--accent-foreground)",
        },
        popover: {
          DEFAULT: "var(--popover)",
          foreground: "var(--popover-foreground)",
        },
        card: {
          DEFAULT: "var(--card)",
          foreground: "var(--card-foreground)",
        },
        // Soft UI 低饱和扩展色彩
        tech: {
          blue: '#5c9ce6',
          purple: '#9d8ce0',
          cyan: '#56b8cf',
          green: '#4cb782',
          red: '#e26d6d',
          orange: '#e8955f',
          yellow: '#d9b44a',
          pink: '#e08bb1',
        },
        glow: {
          blue: 'rgba(92, 156, 230, 0.25)',
          green: 'rgba(76, 183, 130, 0.25)',
          red: 'rgba(226, 109, 109, 0.25)',
          purple: 'rgba(157, 140, 224, 0.25)',
        }
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      backgroundImage: {
        'gradient-primary': 'var(--gradient-primary)',
        'gradient-secondary': 'var(--gradient-secondary)',
        'gradient-accent': 'var(--gradient-accent)',
        'gradient-danger': 'var(--gradient-danger)',
        'tech-grid': 'linear-gradient(rgba(99, 102, 241, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(99, 102, 241, 0.05) 1px, transparent 1px)',
      },
      boxShadow: {
        'glow': 'var(--shadow-glow)',
        'glow-green': 'var(--shadow-glow-green)',
        'glow-red': 'var(--shadow-glow-red)',
        'soft': 'var(--shadow-soft)',
        'soft-lg': 'var(--shadow-soft-lg)',
        'tech': 'var(--shadow-soft)',
        'tech-lg': 'var(--shadow-soft-lg)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-in': 'slideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        'background-shift': 'backgroundShift 20s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: 'var(--shadow-glow)' },
          '100%': { boxShadow: '0 0 30px rgba(59, 130, 246, 0.5)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        backgroundShift: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      backdropBlur: {
        'tech': '10px',
      },
      fontFamily: {
        'tech': ['"Segoe UI"', 'system-ui', '-apple-system', '"PingFang SC"', '"Microsoft YaHei"', 'sans-serif'],
        'mono': ['ui-monospace', '"SF Mono"', '"Cascadia Code"', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
