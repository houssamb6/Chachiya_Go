import { cn } from "@/lib/utils";

interface TunisianPatternProps {
  className?: string;
  variant?: "geometric" | "stars" | "arches";
}

/**
 * Subtle SVG-based Tunisian geometric patterns for backgrounds.
 * Inspired by Islamic geometry & North African tilework — not cliché.
 */
const TunisianPattern = ({ className, variant = "geometric" }: TunisianPatternProps) => {
  if (variant === "stars") {
    return (
      <svg
        className={cn("pointer-events-none select-none", className)}
        viewBox="0 0 200 200"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="none"
      >
        <defs>
          <pattern id="tunisian-stars" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
            {/* 8-pointed star motif */}
            <path
              d="M25 5 L30 20 L45 25 L30 30 L25 45 L20 30 L5 25 L20 20 Z"
              stroke="currentColor"
              strokeWidth="0.5"
              fill="none"
              opacity="0.08"
            />
            <circle cx="25" cy="25" r="2" stroke="currentColor" strokeWidth="0.3" fill="none" opacity="0.06" />
          </pattern>
        </defs>
        <rect width="200" height="200" fill="url(#tunisian-stars)" />
      </svg>
    );
  }

  if (variant === "arches") {
    return (
      <svg
        className={cn("pointer-events-none select-none", className)}
        viewBox="0 0 200 200"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="none"
      >
        <defs>
          <pattern id="tunisian-arches" x="0" y="0" width="60" height="40" patternUnits="userSpaceOnUse">
            {/* Moorish arch */}
            <path
              d="M10 40 L10 20 Q10 5 30 5 Q50 5 50 20 L50 40"
              stroke="currentColor"
              strokeWidth="0.5"
              fill="none"
              opacity="0.07"
            />
            <path
              d="M18 40 L18 22 Q18 12 30 12 Q42 12 42 22 L42 40"
              stroke="currentColor"
              strokeWidth="0.3"
              fill="none"
              opacity="0.05"
            />
          </pattern>
        </defs>
        <rect width="200" height="200" fill="url(#tunisian-arches)" />
      </svg>
    );
  }

  // Default: geometric tessellation
  return (
    <svg
      className={cn("pointer-events-none select-none", className)}
      viewBox="0 0 200 200"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="none"
    >
      <defs>
        <pattern id="tunisian-geo" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
          {/* Interlocking geometric shapes */}
          <rect x="0" y="0" width="40" height="40" stroke="currentColor" strokeWidth="0.3" fill="none" opacity="0.05" />
          <line x1="0" y1="0" x2="40" y2="40" stroke="currentColor" strokeWidth="0.3" opacity="0.06" />
          <line x1="40" y1="0" x2="0" y2="40" stroke="currentColor" strokeWidth="0.3" opacity="0.06" />
          <circle cx="20" cy="20" r="8" stroke="currentColor" strokeWidth="0.4" fill="none" opacity="0.05" />
          <rect x="12" y="12" width="16" height="16" stroke="currentColor" strokeWidth="0.3" fill="none" opacity="0.04" transform="rotate(45 20 20)" />
        </pattern>
      </defs>
      <rect width="200" height="200" fill="url(#tunisian-geo)" />
    </svg>
  );
};

export default TunisianPattern;
