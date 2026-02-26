import { Sparkles, BookOpen, User, Shield, Zap, Brain } from 'lucide-react';

interface Props {
  onSuggestion: (text: string) => void;
}

const suggestions = [
  {
    icon: <BookOpen size={16} />,
    label: 'Knowledge base',
    prompt: 'What is retrieval-augmented generation?',
    gradient: 'from-emerald-500/20 to-teal-500/20',
    border: 'border-emerald-500/20 hover:border-emerald-500/40',
    iconColor: 'text-emerald-400',
  },
  {
    icon: <User size={16} />,
    label: 'About me',
    prompt: "What's my name?",
    gradient: 'from-blue-500/20 to-cyan-500/20',
    border: 'border-blue-500/20 hover:border-blue-500/40',
    iconColor: 'text-blue-400',
  },
  {
    icon: <Sparkles size={16} />,
    label: 'General',
    prompt: 'Explain the difference between SQL and NoSQL databases',
    gradient: 'from-purple-500/20 to-pink-500/20',
    border: 'border-purple-500/20 hover:border-purple-500/40',
    iconColor: 'text-purple-400',
  },
  {
    icon: <Shield size={16} />,
    label: 'Privacy',
    prompt: 'What data do you store about me?',
    gradient: 'from-amber-500/20 to-orange-500/20',
    border: 'border-amber-500/20 hover:border-amber-500/40',
    iconColor: 'text-amber-400',
  },
];

export default function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4 fade-in">
      {/* Logo + Title */}
      <div className="mb-10 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 mb-5">
          <Brain size={28} className="text-accent" />
        </div>
        <h1 className="text-3xl font-bold text-white mb-3 tracking-tight">
          RAG Chat
        </h1>
        <p className="text-sidebar-muted text-sm max-w-lg leading-relaxed">
          Ask me anything. I use retrieval-augmented generation with a private
          knowledge base, topic threading, and research insights — and I remember
          your preferences across conversations.
        </p>
        <div className="flex items-center justify-center gap-4 mt-4 text-[10px] text-sidebar-muted/70">
          <span className="flex items-center gap-1"><Zap size={10} className="text-accent" /> Streaming AI</span>
          <span className="flex items-center gap-1"><Brain size={10} className="text-purple-400" /> Topic Threading</span>
          <span className="flex items-center gap-1"><Sparkles size={10} className="text-amber-400" /> Research Memory</span>
        </div>
      </div>

      {/* Suggestion cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl w-full">
        {suggestions.map((s, i) => (
          <button
            key={s.prompt}
            onClick={() => onSuggestion(s.prompt)}
            className={`stagger-item hover-lift flex items-start gap-3 p-4 rounded-xl border
                       bg-gradient-to-br ${s.gradient} ${s.border}
                       text-left transition-all duration-200 group`}
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <span className={`${s.iconColor} mt-0.5 transition-transform group-hover:scale-110`}>{s.icon}</span>
            <div>
              <div className="text-[10px] text-sidebar-muted mb-1 uppercase tracking-wider font-medium">{s.label}</div>
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
