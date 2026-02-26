/**
 * AI Thread Panel — active thread label, summary, insights.
 *
 * Shows the current conversation's topic threads with their labels,
 * message counts, and associated insights. Fetches from the backend
 * /conversations/{id}/threads endpoint.
 */

import { useEffect } from 'react';
import {
  GitBranch,
  MessageSquare,
  Lightbulb,
  Link2,
  Clock,
  ChevronDown,
} from 'lucide-react';
import { useState } from 'react';
import { useChatStore } from '../../store';

export default function AIThreadPanel() {
  const {
    conversationId,
    threads,
    insights,
    refreshThreads,
    refreshInsights,
  } = useChatStore();

  const [expandedThread, setExpandedThread] = useState<string | null>(null);

  useEffect(() => {
    if (conversationId) {
      refreshThreads(conversationId);
      refreshInsights(conversationId);
    }
  }, [conversationId, refreshThreads, refreshInsights]);

  if (!conversationId) {
    return (
      <div className="p-4 text-center text-sidebar-muted text-xs">
        Start a conversation to see topic threads
      </div>
    );
  }

  if (threads.length === 0) {
    return (
      <div className="p-4 text-center text-sidebar-muted text-xs">
        <GitBranch size={16} className="mx-auto mb-2 opacity-50" />
        No threads yet — threads emerge as you chat
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-sidebar-border/50">
        <GitBranch size={14} className="text-accent" />
        <span className="text-xs font-semibold text-sidebar-text tracking-wide uppercase">
          Topic Threads
        </span>
        <span className="ml-auto text-[10px] text-sidebar-muted bg-sidebar-hover px-1.5 py-0.5 rounded-full">
          {threads.length}
        </span>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto py-2 space-y-1">
        {threads.map((thread) => {
          const threadInsights = insights.filter(
            (i) => i.thread_id === thread.thread_id,
          );
          const isExpanded = expandedThread === thread.thread_id;

          return (
            <div key={thread.thread_id} className="mx-2">
              <button
                onClick={() =>
                  setExpandedThread(isExpanded ? null : thread.thread_id)
                }
                className={`w-full flex items-start gap-2 px-3 py-2.5 rounded-lg text-left
                  transition-colors text-sm ${
                    isExpanded
                      ? 'bg-sidebar-active/50 text-white'
                      : 'text-sidebar-text hover:bg-sidebar-hover'
                  }`}
              >
                <GitBranch
                  size={12}
                  className="mt-0.5 flex-shrink-0 text-accent/70"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium truncate text-xs">
                      {thread.label || thread.thread_id.slice(0, 8) + '…'}
                    </span>
                    <ChevronDown
                      size={10}
                      className={`transition-transform text-sidebar-muted ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-[10px] text-sidebar-muted">
                    <span className="flex items-center gap-1">
                      <MessageSquare size={9} />
                      {thread.message_count} msgs
                    </span>
                    {threadInsights.length > 0 && (
                      <span className="flex items-center gap-1">
                        <Lightbulb size={9} />
                        {threadInsights.length} insights
                      </span>
                    )}
                    {thread.updated_at && (
                      <span className="flex items-center gap-1">
                        <Clock size={9} />
                        {new Date(thread.updated_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
              </button>

              {/* Expanded detail */}
              {isExpanded && (
                <div className="ml-7 mt-1 mb-2 space-y-2 slide-in">
                  {/* Summary */}
                  {thread.summary && (
                    <div className="px-3 py-2 rounded bg-sidebar-hover/30 text-[11px] text-gray-300 leading-relaxed">
                      {thread.summary}
                    </div>
                  )}

                  {/* Thread insights */}
                  {threadInsights.length > 0 && (
                    <div className="space-y-1">
                      <div className="px-1 text-[10px] font-medium text-sidebar-muted uppercase tracking-wider">
                        Insights
                      </div>
                      {threadInsights.map((insight) => (
                        <div
                          key={insight.id}
                          className="flex items-start gap-2 px-3 py-1.5 rounded bg-sidebar-hover/20 text-[10px]"
                        >
                          <InsightTypeIcon type={insight.insight_type} />
                          <div className="flex-1 min-w-0">
                            <span className="text-gray-300">
                              {insight.content}
                            </span>
                            <div className="flex items-center gap-2 mt-0.5 text-sidebar-muted">
                              <span className="capitalize">
                                {insight.insight_type.replace('_', ' ')}
                              </span>
                              {insight.confidence > 0 && (
                                <span>
                                  {Math.round(insight.confidence * 100)}%
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Thread ID */}
                  <div className="px-1 text-[9px] text-sidebar-muted/50 font-mono">
                    {thread.thread_id}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Insight type icons ─────────────────────────────────────────────────── */

function InsightTypeIcon({ type }: { type: string }) {
  const iconClass = 'flex-shrink-0 mt-0.5';
  switch (type) {
    case 'decision':
      return <Lightbulb size={10} className={`${iconClass} text-amber-400`} />;
    case 'conclusion':
      return <Lightbulb size={10} className={`${iconClass} text-emerald-400`} />;
    case 'hypothesis':
      return <Lightbulb size={10} className={`${iconClass} text-purple-400`} />;
    case 'open_question':
      return <Lightbulb size={10} className={`${iconClass} text-blue-400`} />;
    case 'observation':
      return <Lightbulb size={10} className={`${iconClass} text-gray-400`} />;
    default:
      return <Link2 size={10} className={`${iconClass} text-sidebar-muted`} />;
  }
}
