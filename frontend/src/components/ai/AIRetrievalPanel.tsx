/**
 * AI Retrieval Panel — expandable drawer showing RAG sources,
 * Q&A matches, profile data, and full policy decision.
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
      icon: <Route size={12} />,
      label: 'Retrieval Route',
      value: info.route.replace(/_/g, ' '),
      color: 'text-blue-400',
    });
  }

  if (info.intent) {
    sections.push({
      icon: <Eye size={12} />,
      label: 'Intent',
      value: `${info.intent} (${info.confidence != null ? Math.round(info.confidence * 100) + '%' : '?'})`,
      color: 'text-purple-400',
    });
  }

  if (info.num_docs != null) {
    sections.push({
      icon: <FileText size={12} />,
      label: 'Knowledge Base Documents',
      value: `${info.num_docs} chunks retrieved via FAISS`,
      color: 'text-emerald-400',
    });
  }

  if (info.similar_queries != null) {
    sections.push({
      icon: <Database size={12} />,
      label: 'Cross-Conversation Q&A',
      value: `${info.similar_queries} similar past questions matched`,
      color: 'text-amber-400',
    });
  }

  if (info.same_conv_qa != null) {
    sections.push({
      icon: <MessageSquare size={12} />,
      label: 'Same-Conversation Q&A',
      value: `${info.same_conv_qa} relevant exchanges in this thread`,
      color: 'text-cyan-400',
    });
  }

  if (info.profile_injected) {
    sections.push({
      icon: <User size={12} />,
      label: 'Profile Data',
      value: 'User profile was injected into context',
      color: 'text-pink-400',
    });
  }

  if (info.greeting_personalized) {
    sections.push({
      icon: <Sparkles size={12} />,
      label: 'Personalization',
      value: 'Greeting personalized with user\'s name',
      color: 'text-yellow-400',
    });
  }

  if (info.topic_similarity != null) {
    sections.push({
      icon: <ArrowRight size={12} />,
      label: 'Topic Similarity',
      value: `Cosine similarity: ${info.topic_similarity}`,
      color: 'text-zinc-500',
    });
  }

  if (sections.length === 0) return null;

  return (
    <div className="mt-3 rounded-xl glass-subtle overflow-hidden">
      <div className="px-3.5 py-2 border-b border-surface-2/30">
        <span className="text-2xs font-semibold text-zinc-500 tracking-wider uppercase">
          Retrieval & Memory
        </span>
      </div>
      <div className="p-3.5 space-y-3">
        {sections.map((s) => (
          <div key={s.label} className="flex items-start gap-2.5">
            <span className={`mt-0.5 opacity-80 ${s.color}`}>{s.icon}</span>
            <div>
              <div className="text-[11px] font-medium text-zinc-300">{s.label}</div>
              <div className="text-2xs text-zinc-500">{s.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
