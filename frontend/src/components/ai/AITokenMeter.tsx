/**
 * AI Token Meter — visual token / context usage indicator.
 *
 * Shows estimated token usage relative to the model's context window.
 */
import type { CSSProperties } from 'react';

interface Props {
  /** Estimated tokens used in the prompt+response */
  used?: number;
  /** Model context window size */
  limit?: number;
}

const DEFAULT_LIMIT = 65_536; // gpt-oss-120b context window

export default function AITokenMeter({ used, limit = DEFAULT_LIMIT }: Props) {
  if (used == null || used <= 0) return null;

  const pct = Math.min((used / limit) * 100, 100);
  const color =
    pct > 80 ? 'bg-red-500' :
    pct > 50 ? 'bg-yellow-500' :
    'bg-accent';

  return (
    <div className="flex items-center gap-2 text-2xs text-zinc-500 ml-1">
      <span className="font-mono">~{used.toLocaleString()} tok</span>
      <div className="w-14 h-1 rounded-full bg-surface-2 overflow-hidden">
        <div
          className={`h-full rounded-full token-progress-bar ${color}`}
          style={{ '--progress': `${pct}%` } as CSSProperties}
        />
      </div>
      <span className="opacity-40">{pct.toFixed(0)}%</span>
    </div>
  );
}
