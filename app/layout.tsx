import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GEOSPATIALS // Autonomous Mineral Intelligence Workspace',
  description: 'High-precision geospatial intelligence and real-time subsurface telemetry dashboard.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500;1,6..72,600&family=Schibsted+Grotesk:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap"
          rel="stylesheet"
        />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossOrigin="" />
      </head>
      <body className="h-full bg-[#090D16] text-[#F8FAFC] antialiased selection:bg-[#38BDF8]/30 overflow-x-hidden font-sans">
        {children}
      </body>
    </html>
  );
}
