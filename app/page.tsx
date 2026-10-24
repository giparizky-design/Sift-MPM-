'use client';

import dynamic from 'next/dynamic';

// Dynamic import with SSR disabled to prevent Leaflet / window object hydration errors
const GeospatialWorkspace = dynamic(
  () => import('../ui/GeospatialWorkspace'),
  { ssr: false }
);

export default function Page() {
  return (
    <main className="min-h-screen w-full bg-[#090D16]">
      <GeospatialWorkspace />
    </main>
  );
}
