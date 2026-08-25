/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Use environment variable for API URL, fallback to your active Cloudflare tunnel URL for production
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://modes-widespread-informative-shaped.trycloudflare.com';
    
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        source: '/docs',
        destination: `${apiUrl}/docs`,
      },
      {
        source: '/health',
        destination: `${apiUrl}/health`,
      },
    ];
  },
};

export default nextConfig;