/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  output: 'standalone',
  // 前端自身无 /api 路由：将 /api/* 服务端转发到后端（docker 网络内服务名 backend），
  // 使 SSH 检测/命令检查/任务轮询在同源下可达。GHX_BACKEND_ORIGIN 可覆盖。
  async rewrites() {
    const backend = process.env.GHX_BACKEND_ORIGIN || 'http://backend:5000'
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }]
  },
}

export default nextConfig