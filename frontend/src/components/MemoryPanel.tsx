import { Database, FileText, MessageSquare, User, ArrowRight } from 'lucide-react';

interface Props {
  info: Record<string, any>;
}

export default function MemoryPanel({ info }: Props) {
  const items: { icon: React.ReactNode; label: string; value: string }[] = [];

  if (info.route) {
    items.push({
      icon: <ArrowRight size={12} />,
      label: 'Route',
      value: info.route,
    });
  }

  if (info.num_docs != null) {
    items.push({
      icon: <FileText size={12} />,
      label: 'Docs retrieved',
      value: String(info.num_docs),
    });
  }

  if (info.similar_queries != null) {
    items.push({
      icon: <Database size={12} />,
      label: 'Similar Q&A',
      value: String(info.similar_queries),
    });
  }

  if (info.same_conv_qa != null) {
    items.push({
      icon: <MessageSquare size={12} />,
      label: 'Conv Q&A',
      value: String(info.same_conv_qa),
    });
  }

  if (info.profile_injected != null) {
    items.push({
      icon: <User size={12} />,
      label: 'Profile injected',
      value: info.profile_injected ? 'Yes' : 'No',
    });
  }

  if (info.greeting_personalized) {
    items.push({
      icon: <User size={12} />,
      label: 'Greeting',
      value: 'Personalized',
    });
  }

  if (info.topic_similarity != null) {
    items.push({
      icon: <Database size={12} />,
      label: 'Topic similarity',
      value: String(info.topic_similarity),
    });
  }

  if (items.length === 0) return null;

  return (
    <div className="mt-2 p-3 rounded-lg bg-sidebar-bg border border-sidebar-border text-xs fade-in">
      <div className="text-sidebar-muted font-medium mb-2">Memory & Retrieval</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        {items.map((item) => (
          <div key={item.label} className="flex items-center gap-1.5 text-sidebar-muted">
            {item.icon}
            <span>{item.label}:</span>
            <span className="text-sidebar-text font-medium">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
