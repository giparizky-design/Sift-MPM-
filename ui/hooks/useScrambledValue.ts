import { useState, useEffect, useRef } from 'react';

/**
 * Reusable Glyph-Scramble & Spring Telemetry Transition Hook
 * 
 * Triggers a crisp 180ms monospace glyph-scramble / decrypt transition
 * when numerical KPIs, coordinates, or telemetry parameters change.
 */
export function useScrambledValue(
  targetValue: string | number,
  durationMs: number = 180
): string {
  const [displayValue, setDisplayValue] = useState<string>(String(targetValue));
  const chars = '0123456789ABCDEF!#%&*+=~◈';
  const animRef = useRef<number | null>(null);

  useEffect(() => {
    const targetStr = String(targetValue);
    const startStr = displayValue;
    const startTime = performance.now();
    const length = Math.max(targetStr.length, startStr.length);

    const updateFrame = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);

      // Determine how many characters are locked to final target
      const resolvedChars = Math.floor(progress * length);

      let scrambled = '';
      for (let i = 0; i < targetStr.length; i++) {
        if (i < resolvedChars) {
          scrambled += targetStr[i];
        } else if (targetStr[i] === ' ' || targetStr[i] === '-' || targetStr[i] === '.' || targetStr[i] === ',') {
          scrambled += targetStr[i];
        } else {
          scrambled += chars[Math.floor(Math.random() * chars.length)];
        }
      }

      setDisplayValue(scrambled);

      if (progress < 1) {
        animRef.current = requestAnimationFrame(updateFrame);
      } else {
        setDisplayValue(targetStr);
      }
    };

    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(updateFrame);

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [targetValue, durationMs]);

  return displayValue;
}

// Backward-compatible alias
export const useGlyphScramble = useScrambledValue;
