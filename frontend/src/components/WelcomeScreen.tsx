import { motion } from 'framer-motion';
import { Sparkles, BookOpen, User, Shield, Zap, Brain, Layers, Search, GitBranch } from 'lucide-react';

interface Props {
  onSuggestion: (text: string) => void;
}

const suggestions = [
  {
    icon: <BookOpen size={18} />,
    label: 'Knowledge',
    prompt: 'What is retrieval-augmented generation?',
    color: 'from-emerald-500 to-teal-400',
    bg: 'bg-emerald-500/5',
    border: 'border-emerald-500/10 hover:border-emerald-400/40',
    iconBg: 'bg-emerald-500/10',
    iconColor: 'text-emerald-400',
    glow: 'hover:shadow-[0_0_30px_-4px_rgba(16,185,129,0.2)]',
  },
  {
    icon: <User size={18} />,
    label: 'Memory',
    prompt: "What's my name?",
    color: 'from-blue-500 to-cyan-400',
    bg: 'bg-blue-500/5',
    border: 'border-blue-500/10 hover:border-blue-400/40',
    iconBg: 'bg-blue-500/10',
    iconColor: 'text-blue-400',
    glow: 'hover:shadow-[0_0_30px_-4px_rgba(59,130,246,0.2)]',
  },
  {
    icon: <Sparkles size={18} />,
    label: 'General',
    prompt: 'Explain the difference between SQL and NoSQL databases',
    color: 'from-violet-500 to-purple-400',
    bg: 'bg-violet-500/5',
    border: 'border-violet-500/10 hover:border-violet-400/40',
    iconBg: 'bg-violet-500/10',
    iconColor: 'text-violet-400',
    glow: 'hover:shadow-[0_0_30px_-4px_rgba(139,92,246,0.2)]',
  },
  {
    icon: <Shield size={18} />,
    label: 'Privacy',
    prompt: 'What data do you store about me?',
    color: 'from-amber-500 to-orange-400',
    bg: 'bg-amber-500/5',
    border: 'border-amber-500/10 hover:border-amber-400/40',
    iconBg: 'bg-amber-500/10',
    iconColor: 'text-amber-400',
    glow: 'hover:shadow-[0_0_30px_-4px_rgba(245,158,11,0.2)]',
  },
];

const capabilities = [
  { icon: <Zap size={13} />, label: 'Streaming AI', color: 'text-emerald-400', bg: 'bg-emerald-500/8' },
  { icon: <Brain size={13} />, label: 'Behavior Engine', color: 'text-violet-400', bg: 'bg-violet-500/8' },
  { icon: <GitBranch size={13} />, label: 'Topic Threading', color: 'text-cyan-400', bg: 'bg-cyan-500/8' },
  { icon: <Search size={13} />, label: 'Semantic Retrieval', color: 'text-amber-400', bg: 'bg-amber-500/8' },
  { icon: <Layers size={13} />, label: 'Multi-Tier Memory', color: 'text-rose-400', bg: 'bg-rose-500/8' },
];

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.07, delayChildren: 0.3 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20, scale: 0.96 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

export default function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
      className="flex flex-col items-center justify-center h-full px-4 relative"
    >
      {/* Decorative ambient orbs */}
      <div className="absolute top-1/4 left-1/3 w-[400px] h-[400px] rounded-full bg-emerald-500/[0.03] blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/3 w-[300px] h-[300px] rounded-full bg-violet-500/[0.03] blur-[80px] pointer-events-none" />

      {/* Logo + Title */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="mb-12 text-center"
      >
        {/* Animated logo */}
        <div className="relative inline-flex items-center justify-center w-20 h-20 rounded-3xl mb-7">
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-emerald-500/20 via-cyan-500/10 to-violet-500/20 animate-breathe" />
          <div className="absolute inset-[1px] rounded-3xl bg-chat-bg" />
          <div className="relative flex items-center justify-center">
            <Brain size={32} className="text-emerald-400" />
          </div>
          {/* Orbiting dot */}
          <div className="absolute inset-0 animate-[spin_8s_linear_infinite]">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-emerald-400/60 blur-[2px]" />
          </div>
        </div>

        <h1 className="text-4xl font-bold text-white mb-3 tracking-tight">
          <span className="text-gradient">Layered Memory</span>
        </h1>
        <p className="text-zinc-500 text-[15px] max-w-lg leading-relaxed mx-auto">
          A behavior-adaptive retrieval architecture with structured cognition,
          topic threading, and research insights — built to remember.
        </p>

        {/* Capability pills */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="flex flex-wrap items-center justify-center gap-2 mt-6"
        >
          {capabilities.map((cap) => (
            <span
              key={cap.label}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-2xs font-medium
                         ${cap.bg} ${cap.color} border border-white/[0.04]
                         transition-all duration-300 hover:scale-105 hover:border-white/10`}
            >
              {cap.icon}
              {cap.label}
            </span>
          ))}
        </motion.div>
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
            whileHover={{ y: -4, transition: { duration: 0.2 } }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSuggestion(s.prompt)}
            className={`relative flex items-start gap-3.5 p-4 rounded-2xl border overflow-hidden
                       ${s.bg} ${s.border} text-left transition-all duration-300 group ${s.glow}`}
          >
            {/* Subtle gradient shimmer on hover */}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
                            bg-gradient-to-br ${s.color} mix-blend-soft-light pointer-events-none`}
              style={{ opacity: 0 }}
            />

            <span className={`relative flex-shrink-0 p-2 rounded-xl ${s.iconBg} ${s.iconColor}
                            transition-all duration-300 group-hover:scale-110`}>
              {s.icon}
            </span>
            <div className="relative">
              <div className="text-2xs text-zinc-500 mb-1 uppercase tracking-wider font-semibold">
                {s.label}
              </div>
              <div className="text-[13px] text-zinc-400 group-hover:text-zinc-200 transition-colors duration-300 leading-relaxed">
                {s.prompt}
              </div>
            </div>
          </motion.button>
        ))}
      </motion.div>
    </motion.div>
  );
}
