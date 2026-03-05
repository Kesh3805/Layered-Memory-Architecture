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
import { motion, AnimatePresence } from 'framer-motion';
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
  decision: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  conclusion: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  hypothesis: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  open_question: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  observation: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20',
};

const overlayVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
};

const panelVariants = {
  hidden: { opacity: 0, scale: 0.96, y: 12 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
  exit: {
    opacity: 0,
    scale: 0.96,
    y: 12,
    transition: { duration: 0.15 },
  },
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
    <AnimatePresence>
      <motion.div
        key="research-overlay"
        variants={overlayVariants}
        initial="hidden"
        animate="visible"
        exit="hidden"
        className="fixed inset-0 z-50 flex items-center justify-center modal-backdrop"
      >
        <motion.div
          variants={panelVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          className="w-[900px] max-w-[95vw] h-[700px] max-h-[90vh] glass border border-surface-2/40
                     rounded-2xl shadow-elevated-lg flex flex-col overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center gap-3 px-5 py-4 border-b border-surface-2/30
                          bg-gradient-to-r from-surface-1/50 to-transparent">
            <div className="p-1.5 rounded-lg bg-accent/10">
              <Brain size={16} className="text-accent" />
            </div>
            <h2 className="text-sm font-semibold text-white tracking-tight">
              Research Dashboard
            </h2>
            <div className="ml-auto flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-2xs text-zinc-500">
                <GitBranch size={10} />
                {threads.length} threads
                <span className="mx-0.5 text-zinc-700">·</span>
                <Lightbulb size={10} />
                {insights.length} insights
                <span className="mx-0.5 text-zinc-700">·</span>
                <Link2 size={10} />
                {conceptGraph.length} concepts
              </div>
              <button
                onClick={toggleResearchDash}
                title="Close research dashboard"
                className="p-1.5 rounded-lg text-zinc-500 hover:text-white hover:bg-surface-1
                           transition-all duration-200"
              >
                <X size={15} />
              </button>
            </div>
          </div>

          {/* Tab bar */}
          <div className="flex gap-0.5 px-5 pt-3 pb-0">
            {(
              [
                { id: 'threads', label: 'Threads', icon: <GitBranch size={11} /> },
                { id: 'insights', label: 'Insights', icon: <Lightbulb size={11} /> },
                { id: 'concepts', label: 'Concept Graph', icon: <Link2 size={11} /> },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative flex items-center gap-1.5 px-3.5 py-2.5 text-xs font-medium
                  rounded-t-xl transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'text-white bg-surface-1/50'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
              >
                {tab.icon}
                {tab.label}
                {activeTab === tab.id && (
                  <motion.div
                    layoutId="tab-underline"
                    className="absolute bottom-0 left-3 right-3 h-[2px] bg-accent rounded-full"
                  />
                )}
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
        </motion.div>
      </motion.div>
    </AnimatePresence>
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
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <GitBranch size={28} className="mb-3 opacity-20" />
        <p className="text-sm text-zinc-400">No threads yet</p>
        <p className="text-2xs mt-1">Threads emerge as conversations develop topics</p>
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
            className="rounded-xl border border-surface-2/30 bg-surface-1/30 p-4 space-y-2.5
                       hover:border-surface-3/40 transition-colors duration-200"
          >
            <div className="flex items-center gap-2">
              <GitBranch size={12} className="text-accent" />
              <span className="text-sm font-medium text-white tracking-tight">
                {thread.label || thread.id.slice(0, 12) + '…'}
              </span>
            </div>
            <div className="flex items-center gap-3 text-2xs text-zinc-500">
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
              <p className="text-[11px] text-zinc-400 leading-relaxed">
                {thread.summary}
              </p>
            )}
            {threadInsights.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {threadInsights.slice(0, 3).map((i) => (
                  <span
                    key={i.id}
                    className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[9px] border ${
                      INSIGHT_COLORS[i.insight_type] || INSIGHT_COLORS.observation
                    }`}
                  >
                    {i.insight_type.replace('_', ' ')}
                  </span>
                ))}
                {threadInsights.length > 3 && (
                  <span className="text-[9px] text-zinc-600">
                    +{threadInsights.length - 3} more
                  </span>
                )}
              </div>
            )}
            <div className="text-[9px] text-zinc-700 font-mono">
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
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 max-w-xs">
          <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-600" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search insights…"
            className="w-full pl-8 pr-3 py-2 text-xs rounded-xl bg-surface-1/50 border border-surface-2/30
                       text-zinc-300 placeholder:text-zinc-600 focus:outline-none focus:border-accent/30
                       transition-colors duration-200"
          />
        </div>
        <div className="flex items-center gap-1">
          <Filter size={10} className="text-zinc-600 mr-1" />
          {['all', 'decision', 'conclusion', 'hypothesis', 'open_question', 'observation'].map(
            (type) => (
              <button
                key={type}
                onClick={() => setInsightFilter(type)}
                className={`px-2.5 py-1 text-2xs rounded-full border transition-all duration-200 ${
                  insightFilter === type
                    ? 'bg-accent/10 border-accent/30 text-accent shadow-glow-sm'
                    : 'border-surface-2/30 text-zinc-500 hover:text-zinc-300 hover:border-surface-3/40'
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
        <div className="flex flex-col items-center justify-center py-12 text-zinc-600">
          <Lightbulb size={24} className="mb-3 opacity-20" />
          <p className="text-sm text-zinc-400">No insights found</p>
        </div>
      ) : (
        <div className="space-y-2">
          {insights.map((insight) => (
            <div
              key={insight.id}
              className={`flex items-start gap-3 px-4 py-3 rounded-xl border transition-colors duration-200 ${
                INSIGHT_COLORS[insight.insight_type] || INSIGHT_COLORS.observation
              }`}
            >
              <Lightbulb size={13} className="mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-[11px] leading-relaxed">{insight.insight_text}</p>
                <div className="flex items-center gap-3 mt-1.5 text-2xs opacity-60">
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
      <div className="flex flex-col items-center justify-center py-16 text-zinc-600">
        <Link2 size={28} className="mb-3 opacity-20" />
        <p className="text-sm text-zinc-400">No concepts linked yet</p>
        <p className="text-2xs mt-1">Concepts are extracted as threads develop</p>
      </div>
    );
  }

  const maxCount = Math.max(...conceptGraph.map((c) => c.count));

  return (
    <div className="space-y-5">
      {/* Summary stats */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-surface-1/30
                        border border-surface-2/30">
          <BarChart3 size={12} className="text-accent" />
          <span className="text-xs text-zinc-300">
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
                         border border-surface-2/30 bg-surface-1/20
                         hover:bg-accent/8 hover:border-accent/25 transition-all duration-200 cursor-default"
              style={{ fontSize: `${size}px` }}
            >
              <Link2 size={size - 1} className="text-accent/50" />
              <span className="text-zinc-300 font-medium">{concept}</span>
              <span className="text-[9px] text-zinc-600">
                ×{count}
              </span>

              {/* Tooltip */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1.5
                              rounded-lg bg-surface-0 border border-surface-2/40 text-2xs text-zinc-300
                              opacity-0 group-hover:opacity-100 transition-opacity duration-200
                              pointer-events-none whitespace-nowrap z-10 shadow-elevated">
                {count} occurrence{count !== 1 ? 's' : ''} across {threadCount} thread{threadCount !== 1 ? 's' : ''}
              </div>
            </div>
          );
        })}
      </div>

      {/* Concept-to-thread matrix */}
      <div className="mt-4">
        <h3 className="text-2xs font-semibold text-zinc-500 mb-2.5 uppercase tracking-wider">
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
                  className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-surface-1/20
                             text-[11px] text-zinc-300 border border-surface-2/20
                             hover:border-surface-2/40 transition-colors duration-200"
                >
                  <Link2 size={11} className="text-accent flex-shrink-0" />
                  <span className="font-medium">{concept}</span>
                  <span className="text-zinc-600">→</span>
                  <div className="flex items-center gap-1 flex-wrap">
                    {linkedThreads.map((label) => (
                      <span
                        key={label}
                        className="px-2 py-0.5 rounded-md bg-surface-1/40 border border-surface-2/30 text-[9px] text-zinc-400"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          {conceptGraph.filter((c) => c.threadCount > 1).length === 0 && (
            <p className="text-[11px] text-zinc-600 px-3 py-2">
              No cross-thread concept links yet — keep chatting across topics
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
