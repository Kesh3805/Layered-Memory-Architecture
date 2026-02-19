/**
 * AI Retrieval Panel — expandable drawer showing RAG sources,
 * Q&A matches, profile data, and full policy decision.
 *
 * This is the "deep inspection" view, triggered from status bar chips
 * or the Memory button on each message.
 */

import {
  FileText,
  Database,
  MessageSquare,
  User,
  ArrowRight,
  Route,
  Eye,
  Sparkles,
} from 'lucide-react';

interface Props {
  info: Record<string, any>;
}

export default function AIRetrievalPanel({ info }: Props) {
  const sections: { icon: React.ReactNode; label: string; value: string; color: string }[] = [];

  if (info.route) {
    sections.push({
      icon: <Route size={13} />,
      label: 'Retrieval Route',
      value: info.route.replace(/_/g, ' '),
      color: 'text-blue-400',
    });
  }

  if (info.intent) {
    sections.push({
      icon: <Eye size={13} />,
      label: 'Intent',
      value: `${info.intent} (${info.confidence != null ? Math.round(info.confidence * 100) + '%' : '?'})`,
      color: 'text-purple-400',
    });
  }

  if (info.num_docs != null) {
    sections.push({
      icon: <FileText size={13} />,
      label: 'Knowledge Base Documents',
      value: `${info.num_docs} chunks retrieved via FAISS`,
      color: 'text-emerald-400',
    });
  }

  if (info.similar_queries != null) {
    sections.push({
      icon: <Database size={13} />,
      label: 'Cross-Conversation Q&A',
      value: `${info.similar_queries} similar past questions matched`,
      color: 'text-amber-400',
    });
  }

  if (info.same_conv_qa != null) {
    sections.push({
      icon: <MessageSquare size={13} />,
      label: 'Same-Conversation Q&A',
      value: `${info.same_conv_qa} relevant exchanges in this thread`,
      color: 'text-cyan-400',
    });
  }

  if (info.profile_injected) {
    sections.push({
      icon: <User size={13} />,
      label: 'Profile Data',
      value: 'User profile was injected into context',
      color: 'text-pink-400',
    });
  }

  if (info.greeting_personalized) {
    sections.push({
      icon: <Sparkles size={13} />,
      label: 'Personalization',
      value: 'Greeting personalized with user\'s name',
      color: 'text-yellow-400',
    });
  }

  if (info.topic_similarity != null) {
    sections.push({
      icon: <ArrowRight size={13} />,
      label: 'Topic Similarity',
      value: `Cosine similarity: ${info.topic_similarity}`,
      color: 'text-gray-400',
    });
  }

  if (sections.length === 0) return null;

  return (
    <div className="mt-2 rounded-lg bg-[#1a1a1a] border border-sidebar-border overflow-hidden fade-in">
      <div className="px-3 py-2 bg-sidebar-bg/60 border-b border-sidebar-border">
        <span className="text-[11px] font-semibold text-sidebar-muted tracking-wide uppercase">
          Retrieval & Memory
        </span>
      </div>
      <div className="p-3 space-y-2.5">
        {sections.map((s) => (
          <div key={s.label} className="flex items-start gap-2.5">
            <span className={`mt-0.5 ${s.color}`}>{s.icon}</span>
            <div>
              <div className="text-[11px] font-medium text-sidebar-text">{s.label}</div>
              <div className="text-[10px] text-sidebar-muted">{s.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
