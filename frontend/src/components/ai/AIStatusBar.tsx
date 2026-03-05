/**
 * AI Status Bar — horizontal event timeline above each assistant message.
 *
 * Renders streaming stages + final metadata as clickable chips.
 * Stages: classified → retrieved → generating → complete
 */

import {
  Brain,
  Search,
  Loader2,
  CheckCircle2,
  FileText,
  Database,
  MessageSquare,
  User,
  Sparkles,
  GitBranch,
} from 'lucide-react';

export interface StageEvent {
  stage: string;
  intent?: string;
  confidence?: number;
  retrieval_info?: Record<string, any>;
  thread_resolution?: Record<string, any>;
}

interface Props {
  stages: StageEvent[];
  isStreaming: boolean;
  onChipClick?: () => void;
}

const stageIcons: Record<string, React.ReactNode> = {
  classified:  <Brain size={10} />,
  threaded:    <GitBranch size={10} />,
  retrieved:   <Search size={10} />,
  generating:  <Loader2 size={10} className="animate-spin" />,
  complete:    <CheckCircle2 size={10} />,
};

export default function AIStatusBar({ stages, isStreaming, onChipClick }: Props) {
  if (stages.length === 0) return null;

  const retrievedStage = stages.find(s => s.stage === 'retrieved');
  const classifiedStage = stages.find(s => s.stage === 'classified');
  const threadedStage = stages.find(s => s.stage === 'threaded');
  const ri = retrievedStage?.retrieval_info;

  return (
    <div className="flex items-center gap-1 flex-wrap mb-2">
      {classifiedStage && (
        <Chip
          icon={stageIcons.classified}
          label={`Classified: ${classifiedStage.intent}`}
          confidence={classifiedStage.confidence}
          active
        />
      )}

      {threadedStage?.thread_resolution && (
        <Chip
          icon={stageIcons.threaded}
          label={`Threaded: ${threadedStage.thread_resolution.thread_label || threadedStage.thread_resolution.thread_id?.slice(0, 8) + '…'}`}
          active
        />
      )}

      {ri?.num_docs != null && (
        <Chip icon={<FileText size={10} />} label={`Retrieved: ${ri.num_docs} docs`} active />
      )}
      {ri?.similar_queries != null && (
        <Chip icon={<Database size={10} />} label={`Similar Q&A: ${ri.similar_queries}`} active />
      )}
      {ri?.same_conv_qa != null && (
        <Chip icon={<MessageSquare size={10} />} label={`Conv Q&A: ${ri.same_conv_qa}`} active />
      )}
      {ri?.topic_similarity != null && (
        <Chip icon={<Sparkles size={10} />} label={`Topic: ${ri.topic_similarity}`} active />
      )}
      {ri?.profile_injected && (
        <Chip icon={<User size={10} />} label="Profile injected" active />
      )}
      {ri?.greeting_personalized && (
        <Chip icon={<User size={10} />} label="Personalized" active />
      )}

      {isStreaming ? (
        <Chip icon={stageIcons.generating} label="Generating…" active pulse />
      ) : stages.length > 0 && (
        <Chip icon={stageIcons.complete} label="Complete" active />
      )}

      {ri && onChipClick && (
        <button
          onClick={onChipClick}
          className="ml-0.5 px-2 py-0.5 text-2xs rounded-full border border-surface-2
                     text-zinc-500 hover:text-zinc-300 hover:border-accent/30
                     hover:bg-accent/5 transition-all duration-200"
        >
          Details ›
        </button>
      )}
    </div>
  );
}

/* ── Reusable chip ─────────────────────────────────────────────────────── */

function Chip({
  icon,
  label,
  confidence,
  active,
  pulse,
}: {
  icon: React.ReactNode;
  label: string;
  confidence?: number;
  active?: boolean;
  pulse?: boolean;
}) {
  return (
    <span
      className={`ai-chip inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs font-medium
        border transition-all duration-200
        ${active
          ? 'bg-surface-1/80 border-surface-2 text-zinc-400'
          : 'bg-transparent border-surface-2/50 text-zinc-500'}
        ${pulse ? 'animate-pulse' : ''}`}
    >
      {icon}
      {label}
      {confidence != null && (
        <span className="opacity-40">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  );
}
