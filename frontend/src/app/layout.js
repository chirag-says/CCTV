import "./globals.css";

export const metadata = {
  title: "SentinelAI — AI-Powered CCTV Surveillance",
  description: "Production-grade AI-powered video surveillance with face detection, recognition, entry/exit tracking, and analytics.",
  keywords: "CCTV, surveillance, face recognition, AI, security",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236366f1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>" />
      </head>
      <body>{children}</body>
    </html>
  );
}
