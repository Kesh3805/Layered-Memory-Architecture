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
  general:        { icon: <Sparkles size={10} />,       label: 'General',        color: 'bg-blue-500/15 text-blue-400 border-blue-500/20' },
  knowledge_base: { icon: <BookOpen size={10} />,       label: 'Knowledge Base', color: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' },
  continuation:   { icon: <MessageCircle size={10} />,  label: 'Continuation',   color: 'bg-purple-500/15 text-purple-400 border-purple-500/20' },
  profile:        { icon: <User size={10} />,           label: 'Profile',        color: 'bg-amber-500/15 text-amber-400 border-amber-500/20' },
  privacy:        { icon: <Shield size={10} />,         label: 'Privacy',        color: 'bg-red-500/15 text-red-400 border-red-500/20' },
};

function confidenceColor(c: number): string {
  if (c >= 0.85) return 'bg-green-500';
  if (c >= 0.6)  return 'bg-yellow-500';
  return 'bg-gray-500';
}

export default function AIIntentBadge({ intent, confidence, compact }: Props) {
  const cfg = intentConfig[intent] ?? { icon: <Brain size={10} />, label: intent, color: 'bg-sidebar-hover text-sidebar-muted border-sidebar-border' };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium border ${cfg.color}`}>
      {/* Confidence dot */}
      <span className={`w-1.5 h-1.5 rounded-full ${confidenceColor(confidence)}`} />
      {cfg.icon}
      {!compact && <span>{cfg.label}</span>}
      <span className="opacity-60">{Math.round(confidence * 100)}%</span>
    </span>
  );
}
