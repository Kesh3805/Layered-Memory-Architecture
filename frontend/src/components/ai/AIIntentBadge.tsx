/**
 * AI Intent Badge — color-coded indicator for classifier confidence.
 *
 * Green (≥0.85)  — high confidence
 * Yellow (0.6–0.84) — moderate
 * Gray (<0.6) — low / uncertain
 */

import { Brain, Sparkles, BookOpen, User, Shield, MessageCircle } from 'lucide-react';

interface Props {
  intent: string;
  confidence: number;
  compact?: boolean;
}

const intentConfig: Record<string, { icon: React.ReactNode; label: string; color: string }> = {
  general:        { icon: <Sparkles size={9} />,       label: 'General',        color: 'bg-blue-500/8 text-blue-400 border-blue-500/15' },
  knowledge_base: { icon: <BookOpen size={9} />,       label: 'Knowledge Base', color: 'bg-emerald-500/8 text-emerald-400 border-emerald-500/15' },
  continuation:   { icon: <MessageCircle size={9} />,  label: 'Continuation',   color: 'bg-purple-500/8 text-purple-400 border-purple-500/15' },
  profile:        { icon: <User size={9} />,           label: 'Profile',        color: 'bg-amber-500/8 text-amber-400 border-amber-500/15' },
  privacy:        { icon: <Shield size={9} />,         label: 'Privacy',        color: 'bg-red-500/8 text-red-400 border-red-500/15' },
};

function confidenceColor(c: number): string {
  if (c >= 0.85) return 'bg-emerald-500';
  if (c >= 0.6)  return 'bg-yellow-500';
  return 'bg-zinc-500';
}

export default function AIIntentBadge({ intent, confidence, compact }: Props) {
  const cfg = intentConfig[intent] ?? { icon: <Brain size={9} />, label: intent, color: 'bg-surface-1 text-zinc-500 border-surface-2' };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-2xs font-medium border ${cfg.color}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${confidenceColor(confidence)}`} />
      {cfg.icon}
      {!compact && <span>{cfg.label}</span>}
      <span className="opacity-50">{Math.round(confidence * 100)}%</span>
    </span>
  );
}
