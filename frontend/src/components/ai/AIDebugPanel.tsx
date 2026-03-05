/**
 * AI Debug Panel — raw system internals visible when debug mode is on.
 *
 * Shows: PolicyDecision JSON, ThreadResolution JSON, research context,
 * injected prompt frames, raw retrieval info.
 */

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

interface Props {
  metadata: Record<string, any>;
}

export default function AIDebugPanel({ metadata }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (key: string) =>
    setExpanded((s) => ({ ...s, [key]: !s[key] }));

  const sections = [
    { key: 'policy_decision', label: 'PolicyDecision', data: metadata.policy_decision },
    { key: 'thread_resolution', label: 'ThreadResolution', data: metadata.thread_resolution },
    { key: 'research_context', label: 'Research Context', data: metadata.research_context },
    { key: 'retrieval_info', label: 'Retrieval Info', data: metadata.retrieval_info },
    { key: 'query_tags', label: 'Query Tags', data: metadata.query_tags },
    { key: 'full_meta', label: 'Full Metadata', data: metadata },
  ].filter(s => s.data != null);

  return (
    <div className="mt-3 rounded-xl overflow-hidden border border-yellow-500/15 bg-surface-0/80">
      <div className="px-3.5 py-2 bg-yellow-500/5 border-b border-yellow-500/15 flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
        <span className="text-2xs font-semibold text-yellow-400/80 tracking-wider uppercase">
          Debug Mode
        </span>
      </div>
      <div className="divide-y divide-surface-2/30">
        {sections.map(({ key, label, data }) => (
          <div key={key}>
            <button
              onClick={() => toggle(key)}
              className="w-full flex items-center justify-between px-3.5 py-2.5
                         text-[11px] font-medium text-zinc-400 hover:bg-surface-1/30
                         transition-colors duration-200"
            >
              {label}
              <ChevronDown
                size={11}
                className={`transition-transform duration-200 text-zinc-500 ${expanded[key] ? 'rotate-180' : ''}`}
              />
            </button>
            {expanded[key] && (
              <pre className="px-3.5 pb-3 debug-mono text-zinc-500 overflow-x-auto whitespace-pre-wrap">
                {JSON.stringify(data, null, 2)}
              </pre>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
