/**
 * AI Research Dashboard — threads, insights, concept graph.
 *
 * Provides a full-page overlay showing:
 *  - Thread overview with message counts and summaries
 *  - All extracted insights categorized by type
 *  - Concept graph showing cross-thread concept links
 *
 * Fetches from /conversations/{id}/threads, /insights, /concepts endpoints.
 */

import { useEffect, useState, useMemo } from 'react';
import {
  X,
  GitBranch,
  Lightbulb,
  Link2,
  Search,
  Filter,
  BarChart3,
  Brain,
  MessageSquare,
} from 'lucide-react';
import { useChatStore } from '../../store';
import type { Thread, Insight, ConceptLink } from '../../api';

type Tab = 'threads' | 'insights' | 'concepts';

const INSIGHT_COLORS: Record<string, string> = {
  decision: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  conclusion: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  hypothesis: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  open_question: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  observation: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
};

export default function AIResearchDashboard() {
  const {
    conversationId,
    threads,
    insights,
    concepts,
    refreshThreads,
    refreshInsights,
    refreshConcepts,
    researchDashOpen,
    toggleResearchDash,
  } = useChatStore();

  const [activeTab, setActiveTab] = useState<Tab>('threads');
  const [insightFilter, setInsightFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (researchDashOpen && conversationId) {
      refreshThreads(conversationId);
      refreshInsights(conversationId);
      refreshConcepts(conversationId);
    }
  }, [researchDashOpen, conversationId, refreshThreads, refreshInsights, refreshConcepts]);

  const filteredInsights = useMemo(() => {
    let result = insights;
    if (insightFilter !== 'all') {
      result = result.filter((i) => i.insight_type === insightFilter);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((i) => i.insight_text.toLowerCase().includes(q));
    }
    return result;
  }, [insights, insightFilter, searchQuery]);

  // Build concept graph data: group concepts and count connections
  const conceptGraph = useMemo(() => {
    const map = new Map<string, { count: number; threads: Set<string> }>();
    for (const c of concepts) {
      const entry = map.get(c.concept) || { count: 0, threads: new Set() };
      entry.count++;
      if (c.thread_id) entry.threads.add(c.thread_id);
      map.set(c.concept, entry);
    }
    return Array.from(map.entries())
      .map(([concept, data]) => ({
        concept,
        count: data.count,
        threadCount: data.threads.size,
      }))
      .sort((a, b) => b.count - a.count);
  }, [concepts]);

  // Insight type counts for filter
  const insightCounts = useMemo(() => {
    const counts: Record<string, number> = { all: insights.length };
    for (const i of insights) {
      counts[i.insight_type] = (counts[i.insight_type] || 0) + 1;
    }
    return counts;
  }, [insights]);

  if (!researchDashOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center palette-backdrop bg-black/60">
      <div className="w-[900px] max-w-[95vw] h-[700px] max-h-[90vh] bg-sidebar-bg border border-sidebar-border rounded-xl shadow-2xl flex flex-col overflow-hidden scale-in">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-sidebar-border">
          <Brain size={18} className="text-accent" />
          <h2 className="text-sm font-semibold text-white">Research Dashboard</h2>
          <div className="ml-auto flex items-center gap-3">
            <div className="flex items-center gap-1 text-[10px] text-sidebar-muted">
              <GitBranch size={11} />
              {threads.length} threads
              <span className="mx-1">·</span>
              <Lightbulb size={11} />
              {insights.length} insights
              <span className="mx-1">·</span>
              <Link2 size={11} />
              {conceptGraph.length} concepts
            </div>
            <button
              onClick={toggleResearchDash}
              title="Close research dashboard"
              className="p-1.5 rounded-lg text-sidebar-muted hover:text-white hover:bg-sidebar-hover transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 px-5 pt-3 pb-0">
          {(
            [
              { id: 'threads', label: 'Threads', icon: <GitBranch size={12} /> },
              { id: 'insights', label: 'Insights', icon: <Lightbulb size={12} /> },
              { id: 'concepts', label: 'Concept Graph', icon: <Link2 size={12} /> },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-t-lg
                transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? 'border-accent text-white bg-sidebar-hover/30'
                    : 'border-transparent text-sidebar-muted hover:text-sidebar-text'
                }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {activeTab === 'threads' && (
            <ThreadsView threads={threads} insights={insights} />
          )}
          {activeTab === 'insights' && (
            <InsightsView
              insights={filteredInsights}
              insightCounts={insightCounts}
              insightFilter={insightFilter}
              setInsightFilter={setInsightFilter}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              threads={threads}
            />
          )}
          {activeTab === 'concepts' && (
            <ConceptGraphView
              conceptGraph={conceptGraph}
              threads={threads}
              concepts={concepts}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Threads View ───────────────────────────────────────────────────────── */

function ThreadsView({
  threads,
  insights,
}: {
  threads: Thread[];
  insights: Insight[];
}) {
  if (threads.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-sidebar-muted">
        <GitBranch size={32} className="mb-3 opacity-30" />
        <p className="text-sm">No threads yet</p>
        <p className="text-xs mt-1">Threads emerge as conversations develop topics</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {threads.map((thread) => {
        const threadInsights = insights.filter(
          (i) => i.thread_id === thread.id,
        );
        return (
          <div
            key={thread.id}
            className="rounded-lg border border-sidebar-border bg-sidebar-hover/20 p-4 space-y-2"
          >
            <div className="flex items-center gap-2">
              <GitBranch size={13} className="text-accent" />
              <span className="text-sm font-medium text-white">
                {thread.label || thread.id.slice(0, 12) + '…'}
              </span>
            </div>
            <div className="flex items-center gap-3 text-[10px] text-sidebar-muted">
              <span className="flex items-center gap-1">
                <MessageSquare size={9} />
                {thread.message_count} messages
              </span>
              <span className="flex items-center gap-1">
                <Lightbulb size={9} />
                {threadInsights.length} insights
              </span>
            </div>
            {thread.summary && (
              <p className="text-[11px] text-gray-400 leading-relaxed">
                {thread.summary}
              </p>
            )}
            {threadInsights.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {threadInsights.slice(0, 3).map((i) => (
                  <span
                    key={i.id}
                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] border ${
                      INSIGHT_COLORS[i.insight_type] || INSIGHT_COLORS.observation
                    }`}
                  >
                    {i.insight_type.replace('_', ' ')}
                  </span>
                ))}
                {threadInsights.length > 3 && (
                  <span className="text-[9px] text-sidebar-muted">
                    +{threadInsights.length - 3} more
                  </span>
                )}
              </div>
            )}
            <div className="text-[9px] text-sidebar-muted/40 font-mono">
              {thread.id}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Insights View ──────────────────────────────────────────────────────── */

function InsightsView({
  insights,
  insightCounts,
  insightFilter,
  setInsightFilter,
  searchQuery,
  setSearchQuery,
  threads,
}: {
  insights: Insight[];
  insightCounts: Record<string, number>;
  insightFilter: string;
  setInsightFilter: (v: string) => void;
  searchQuery: string;
  setSearchQuery: (v: string) => void;
  threads: Thread[];
}) {
  const getThreadLabel = (threadId: string | null) => {
    if (!threadId) return null;
    const t = threads.find((t) => t.id === threadId);
    return t?.label || threadId.slice(0, 8) + '…';
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-sidebar-muted" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search insights…"
            className="w-full pl-7 pr-3 py-1.5 text-xs rounded-lg bg-sidebar-hover border border-sidebar-border
                       text-sidebar-text placeholder:text-sidebar-muted/50 focus:outline-none focus:border-accent/40"
          />
        </div>
        <div className="flex items-center gap-1">
          <Filter size={11} className="text-sidebar-muted" />
          {['all', 'decision', 'conclusion', 'hypothesis', 'open_question', 'observation'].map(
            (type) => (
              <button
                key={type}
                onClick={() => setInsightFilter(type)}
                className={`px-2 py-1 text-[10px] rounded-full border transition-colors ${
                  insightFilter === type
                    ? 'bg-accent/20 border-accent/40 text-accent'
                    : 'border-sidebar-border text-sidebar-muted hover:text-sidebar-text'
                }`}
              >
                {type === 'all' ? 'All' : type.replace('_', ' ')}
                {insightCounts[type] ? ` (${insightCounts[type]})` : ''}
              </button>
            ),
          )}
        </div>
      </div>

      {/* Insights list */}
      {insights.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-sidebar-muted">
          <Lightbulb size={28} className="mb-3 opacity-30" />
          <p className="text-sm">No insights found</p>
        </div>
      ) : (
        <div className="space-y-2">
          {insights.map((insight) => (
            <div
              key={insight.id}
              className={`flex items-start gap-3 px-4 py-3 rounded-lg border ${
                INSIGHT_COLORS[insight.insight_type] || INSIGHT_COLORS.observation
              }`}
            >
              <Lightbulb size={14} className="mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[11px] leading-relaxed">{insight.insight_text}</p>
                <div className="flex items-center gap-3 mt-1.5 text-[10px] opacity-70">
                  <span className="font-medium capitalize">
                    {insight.insight_type.replace('_', ' ')}
                  </span>
                  {insight.confidence_score > 0 && (
                    <span>{Math.round(insight.confidence_score * 100)}% confidence</span>
                  )}
                  {insight.thread_id && (
                    <span className="flex items-center gap-1">
                      <GitBranch size={9} />
                      {getThreadLabel(insight.thread_id)}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Concept Graph View ─────────────────────────────────────────────────── */

function ConceptGraphView({
  conceptGraph,
  threads,
  concepts,
}: {
  conceptGraph: { concept: string; count: number; threadCount: number }[];
  threads: Thread[];
  concepts: ConceptLink[];
}) {
  if (conceptGraph.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-sidebar-muted">
        <Link2 size={32} className="mb-3 opacity-30" />
        <p className="text-sm">No concepts linked yet</p>
        <p className="text-xs mt-1">Concepts are extracted as threads develop</p>
      </div>
    );
  }

  const maxCount = Math.max(...conceptGraph.map((c) => c.count));

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-sidebar-hover/30 border border-sidebar-border">
          <BarChart3 size={13} className="text-accent" />
          <span className="text-xs text-sidebar-text">
            {conceptGraph.length} unique concepts across {threads.length} threads
          </span>
        </div>
      </div>

      {/* Concept bubble cloud */}
      <div className="flex flex-wrap gap-2">
        {conceptGraph.map(({ concept, count, threadCount }) => {
          const intensity = count / maxCount;
          const size = Math.max(11, Math.round(11 + intensity * 5));

          return (
            <div
              key={concept}
              className="group relative inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full
                         border border-sidebar-border bg-sidebar-hover/30
                         hover:bg-accent/10 hover:border-accent/30 transition-all cursor-default"
              style={{ fontSize: `${size}px` }}
            >
              <Link2 size={size - 1} className="text-accent/60" />
              <span className="text-sidebar-text font-medium">{concept}</span>
              <span className="text-[9px] text-sidebar-muted">
                ×{count}
              </span>

              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1
                              rounded bg-black border border-sidebar-border text-[10px] text-gray-300
                              opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none
                              whitespace-nowrap z-10">
                {count} occurrence{count !== 1 ? 's' : ''} across {threadCount} thread{threadCount !== 1 ? 's' : ''}
              </div>
            </div>
          );
        })}
      </div>

      {/* Concept-to-thread matrix */}
      <div className="mt-4">
        <h3 className="text-xs font-semibold text-sidebar-muted mb-2 uppercase tracking-wider">
          Cross-Thread Links
        </h3>
        <div className="space-y-1">
          {conceptGraph
            .filter((c) => c.threadCount > 1)
            .map(({ concept, threadCount }) => {
              const linkedThreads = concepts
                .filter((c) => c.concept === concept && c.thread_id)
                .map((c) => {
                  const t = threads.find((t) => t.id === c.thread_id);
                  return t?.label || c.thread_id?.slice(0, 8) + '…';
                })
                .filter((v, i, a) => a.indexOf(v) === i); // unique

              return (
                <div
                  key={concept}
                  className="flex items-center gap-2 px-3 py-2 rounded bg-sidebar-hover/20
                             text-[11px] text-sidebar-text"
                >
                  <Link2 size={11} className="text-accent flex-shrink-0" />
                  <span className="font-medium">{concept}</span>
                  <span className="text-sidebar-muted">→</span>
                  <div className="flex items-center gap-1 flex-wrap">
                    {linkedThreads.map((label) => (
                      <span
                        key={label}
                        className="px-1.5 py-0.5 rounded bg-sidebar-hover border border-sidebar-border text-[9px]"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          {conceptGraph.filter((c) => c.threadCount > 1).length === 0 && (
            <p className="text-[11px] text-sidebar-muted px-3 py-2">
              No cross-thread concept links yet — keep chatting across topics
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
