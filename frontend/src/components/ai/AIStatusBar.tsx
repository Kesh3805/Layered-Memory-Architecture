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
  classified:  <Brain size={11} />,
  threaded:    <GitBranch size={11} />,
  retrieved:   <Search size={11} />,
  generating:  <Loader2 size={11} className="animate-spin" />,
  complete:    <CheckCircle2 size={11} />,
};

export default function AIStatusBar({ stages, isStreaming, onChipClick }: Props) {
  if (stages.length === 0) return null;

  // Extract retrieval info from the 'retrieved' or final annotation stage
  const retrievedStage = stages.find(s => s.stage === 'retrieved');
  const classifiedStage = stages.find(s => s.stage === 'classified');
  const threadedStage = stages.find(s => s.stage === 'threaded');
  const ri = retrievedStage?.retrieval_info;

  return (
    <div className="flex items-center gap-1.5 flex-wrap mb-2 fade-in">
      {/* Classified chip */}
      {classifiedStage && (
        <Chip
          icon={stageIcons.classified}
          label={`Classified: ${classifiedStage.intent}`}
          confidence={classifiedStage.confidence}
          active
        />
      )}

      {/* Threaded chip */}
      {threadedStage?.thread_resolution && (
        <Chip
          icon={stageIcons.threaded}
          label={`Threaded: ${threadedStage.thread_resolution.thread_label || threadedStage.thread_resolution.thread_id?.slice(0, 8) + '…'}`}
          active
        />
      )}

      {/* Retrieval chips */}
      {ri?.num_docs != null && (
        <Chip icon={<FileText size={11} />} label={`Retrieved: ${ri.num_docs} docs`} active />
      )}
      {ri?.similar_queries != null && (
        <Chip icon={<Database size={11} />} label={`Similar Q&A: ${ri.similar_queries}`} active />
      )}
      {ri?.same_conv_qa != null && (
        <Chip icon={<MessageSquare size={11} />} label={`Conv Q&A: ${ri.same_conv_qa}`} active />
      )}
      {ri?.topic_similarity != null && (
        <Chip icon={<Sparkles size={11} />} label={`Topic: ${ri.topic_similarity}`} active />
      )}
      {ri?.profile_injected && (
        <Chip icon={<User size={11} />} label="Profile injected" active />
      )}
      {ri?.greeting_personalized && (
        <Chip icon={<User size={11} />} label="Personalized" active />
      )}

      {/* Generating / Complete indicator */}
      {isStreaming ? (
        <Chip icon={stageIcons.generating} label="Generating…" active pulse />
      ) : stages.length > 0 && (
        <Chip icon={stageIcons.complete} label="Complete" active />
      )}

      {/* Memory detail toggle */}
      {ri && onChipClick && (
        <button
          onClick={onChipClick}
          className="ml-1 px-2 py-0.5 text-[10px] rounded-full border border-sidebar-border
                     text-sidebar-muted hover:text-white hover:border-accent/40 transition-colors"
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
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium
        border transition-all
        ${active
          ? 'bg-sidebar-hover border-sidebar-border text-sidebar-text'
          : 'bg-transparent border-sidebar-border/50 text-sidebar-muted'}
        ${pulse ? 'animate-pulse' : ''}`}
    >
      {icon}
      {label}
      {confidence != null && (
        <span className="opacity-50">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  );
}
