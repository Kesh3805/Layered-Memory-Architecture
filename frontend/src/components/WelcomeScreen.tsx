import { Sparkles, BookOpen, User, Shield } from 'lucide-react';

interface Props {
  onSuggestion: (text: string) => void;
}

const suggestions = [
  {
    icon: <BookOpen size={16} />,
    label: 'Knowledge base',
    prompt: 'What is retrieval-augmented generation?',
  },
  {
    icon: <User size={16} />,
    label: 'About me',
    prompt: "What's my name?",
  },
  {
    icon: <Sparkles size={16} />,
    label: 'General',
    prompt: 'Explain the difference between SQL and NoSQL databases',
  },
  {
    icon: <Shield size={16} />,
    label: 'Privacy',
    prompt: 'What data do you store about me?',
  },
];

export default function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4 fade-in">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-semibold text-white mb-2">RAG Chat</h1>
        <p className="text-sidebar-muted text-sm max-w-md">
          Ask me anything.  I use retrieval-augmented generation with a private
          knowledge base and remember your preferences across conversations.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl w-full">
        {suggestions.map((s) => (
          <button
            key={s.prompt}
            onClick={() => onSuggestion(s.prompt)}
            className="flex items-start gap-3 p-4 rounded-xl border border-sidebar-border
                       bg-sidebar-bg/50 hover:bg-sidebar-hover text-left transition-colors group"
          >
            <span className="text-accent mt-0.5">{s.icon}</span>
            <div>
              <div className="text-xs text-sidebar-muted mb-1">{s.label}</div>
              <div className="text-sm text-sidebar-text group-hover:text-white transition-colors">
                {s.prompt}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
