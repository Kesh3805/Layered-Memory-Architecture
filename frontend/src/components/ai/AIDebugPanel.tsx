/**
 * AI Debug Panel — raw system internals visible when debug mode is on.
 *
 * Shows: PolicyDecision JSON, injected prompt frames, raw retrieval info.
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
    { key: 'retrieval_info', label: 'Retrieval Info', data: metadata.retrieval_info },
    { key: 'query_tags', label: 'Query Tags', data: metadata.query_tags },
    { key: 'full_meta', label: 'Full Metadata', data: metadata },
  ].filter(s => s.data != null);

  return (
    <div className="mt-2 rounded-lg bg-[#0d0d0d] border border-yellow-500/20 overflow-hidden fade-in">
      <div className="px-3 py-1.5 bg-yellow-500/10 border-b border-yellow-500/20 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
        <span className="text-[10px] font-semibold text-yellow-400 tracking-wide uppercase">
          Debug Mode
        </span>
      </div>
      <div className="divide-y divide-sidebar-border">
        {sections.map(({ key, label, data }) => (
          <div key={key}>
            <button
              onClick={() => toggle(key)}
              className="w-full flex items-center justify-between px-3 py-2
                         text-[11px] font-medium text-sidebar-text hover:bg-sidebar-hover/30
                         transition-colors"
            >
              {label}
              <ChevronDown
                size={12}
                className={`transition-transform text-sidebar-muted ${expanded[key] ? 'rotate-180' : ''}`}
              />
            </button>
            {expanded[key] && (
              <pre className="px-3 pb-2 text-[10px] text-gray-400 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                {JSON.stringify(data, null, 2)}
              </pre>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
