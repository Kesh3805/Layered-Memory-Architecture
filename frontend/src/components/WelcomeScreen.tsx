import { motion } from 'framer-motion';
import { Sparkles, BookOpen, User, Shield, Zap, Brain } from 'lucide-react';

interface Props {
  onSuggestion: (text: string) => void;
}

const suggestions = [
  {
    icon: <BookOpen size={16} />,
    label: 'Knowledge base',
    prompt: 'What is retrieval-augmented generation?',
    gradient: 'from-emerald-500/10 to-teal-500/10',
    border: 'border-emerald-500/10 hover:border-emerald-500/30',
    iconColor: 'text-emerald-400',
    glow: 'hover:shadow-[0_0_20px_-4px_rgba(16,185,129,0.15)]',
  },
  {
    icon: <User size={16} />,
    label: 'About me',
    prompt: "What's my name?",
    gradient: 'from-blue-500/10 to-cyan-500/10',
    border: 'border-blue-500/10 hover:border-blue-500/30',
    iconColor: 'text-blue-400',
    glow: 'hover:shadow-[0_0_20px_-4px_rgba(59,130,246,0.15)]',
  },
  {
    icon: <Sparkles size={16} />,
    label: 'General',
    prompt: 'Explain the difference between SQL and NoSQL databases',
    gradient: 'from-purple-500/10 to-pink-500/10',
    border: 'border-purple-500/10 hover:border-purple-500/30',
    iconColor: 'text-purple-400',
    glow: 'hover:shadow-[0_0_20px_-4px_rgba(168,85,247,0.15)]',
  },
  {
    icon: <Shield size={16} />,
    label: 'Privacy',
    prompt: 'What data do you store about me?',
    gradient: 'from-amber-500/10 to-orange-500/10',
    border: 'border-amber-500/10 hover:border-amber-500/30',
    iconColor: 'text-amber-400',
    glow: 'hover:shadow-[0_0_20px_-4px_rgba(245,158,11,0.15)]',
  },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.2 },
  },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

export default function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col items-center justify-center h-full px-4"
    >
      {/* Logo + Title */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="mb-12 text-center"
      >
        <div className="relative inline-flex items-center justify-center w-16 h-16 rounded-2xl
                        bg-accent/5 border border-accent/10 mb-6">
          <Brain size={28} className="text-accent" />
          {/* Subtle animated ring */}
          <div className="absolute inset-0 rounded-2xl animate-glow-pulse" />
        </div>
        <h1 className="text-3xl font-bold text-white mb-3 tracking-tight">
          <span className="text-gradient">RAG Chat</span>
        </h1>
        <p className="text-zinc-500 text-sm max-w-md leading-relaxed mx-auto">
          Ask me anything. I use retrieval-augmented generation with a private
          knowledge base, topic threading, and research insights — and I remember
          your preferences across conversations.
        </p>
        <div className="flex items-center justify-center gap-6 mt-5">
          <FeatureTag icon={<Zap size={10} />} label="Streaming AI" color="text-accent" />
          <FeatureTag icon={<Brain size={10} />} label="Topic Threading" color="text-purple-400" />
          <FeatureTag icon={<Sparkles size={10} />} label="Research Memory" color="text-amber-400" />
        </div>
      </motion.div>

      {/* Suggestion cards */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl w-full"
      >
        {suggestions.map((s) => (
          <motion.button
            key={s.prompt}
            variants={item}
            whileHover={{ y: -3, transition: { duration: 0.2 } }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onSuggestion(s.prompt)}
            className={`flex items-start gap-3 p-4 rounded-xl border
                       bg-gradient-to-br ${s.gradient} ${s.border}
                       text-left transition-all duration-300 group ${s.glow}`}
          >
            <span className={`${s.iconColor} mt-0.5 transition-transform duration-200
                             group-hover:scale-110`}>
              {s.icon}
            </span>
            <div>
              <div className="text-2xs text-zinc-500 mb-1 uppercase tracking-wider font-semibold">
                {s.label}
              </div>
              <div className="text-[13px] text-zinc-400 group-hover:text-zinc-200 transition-colors duration-200">
                {s.prompt}
              </div>
            </div>
          </motion.button>
        ))}
      </motion.div>
    </motion.div>
  );
}

/* ── Feature tag ────────────────────────────────────────────────────────── */

function FeatureTag({ icon, label, color }: { icon: React.ReactNode; label: string; color: string }) {
  return (
    <span className="flex items-center gap-1.5 text-2xs text-zinc-600">
      <span className={color}>{icon}</span>
      {label}
    </span>
  );
}
