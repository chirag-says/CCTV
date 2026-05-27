/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/snapshots/:path*',
        destination: 'http://localhost:8000/snapshots/:path*',
      },
    ];
  },
};

export default nextConfig;
