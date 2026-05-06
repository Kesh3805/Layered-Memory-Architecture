import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Square, Paperclip, Sparkles } from 'lucide-react';
import type { useChatStream } from '../hooks/use-chat-stream';

interface Props {
  chat: ReturnType<typeof useChatStream>;
}

export default function InputArea({ chat }: Props) {
  const [value, setValue] = useState('');
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const charCount = value.length;

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || chat.isLoading) return;
    chat.send(trimmed);
    setValue('');
    // Reset height
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-white/[0.04] bg-gradient-to-t from-surface-0 via-surface-0/95 to-transparent backdrop-blur-xl">
      <div className="max-w-3xl mx-auto px-4 py-4">
        <div
          className={`relative flex items-end gap-2 rounded-2xl border px-4 py-3
                     transition-all duration-400 ease-out
                     ${focused
                       ? 'border-accent/40 shadow-[0_0_30px_-8px_rgba(16,185,129,0.15)] bg-surface-1/70'
                       : 'border-white/[0.06] hover:border-white/[0.1] bg-surface-1/40'}`}
        >
          {/* Animated gradient border when focused */}
          {focused && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 rounded-2xl pointer-events-none"
              style={{
                background: 'linear-gradient(135deg, rgba(16,185,129,0.06), rgba(6,182,212,0.04), rgba(139,92,246,0.06))',
              }}
            />
          )}

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Ask anything — I'll retrieve, reason, and remember..."
            rows={1}
            className="relative flex-1 bg-transparent outline-none resize-none text-sm text-zinc-200
                       placeholder:text-zinc-600 max-h-[200px] leading-relaxed z-10"
          />

          <div className="relative flex items-center gap-1.5 z-10">
            {/* Character counter (subtle) */}
            {charCount > 0 && (
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-2xs text-zinc-600 tabular-nums mr-1"
              >
                {charCount}
              </motion.span>
            )}

            <AnimatePresence mode="wait">
              {chat.isLoading ? (
                <motion.button
                  key="stop"
                  initial={{ scale: 0.7, opacity: 0, rotate: -90 }}
                  animate={{ scale: 1, opacity: 1, rotate: 0 }}
                  exit={{ scale: 0.7, opacity: 0, rotate: 90 }}
                  transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  onClick={chat.stop}
                  className="flex-shrink-0 p-2.5 rounded-xl bg-zinc-700/50 text-zinc-300
                             hover:bg-zinc-600/60 transition-all duration-200
                             border border-zinc-600/30"
                  title="Stop generating"
                >
                  <Square size={14} />
                </motion.button>
              ) : (
                <motion.button
                  key="send"
                  initial={{ scale: 0.7, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.7, opacity: 0 }}
                  transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  onClick={handleSubmit}
                  disabled={!value.trim()}
                  className="flex-shrink-0 p-2.5 rounded-xl transition-all duration-300
                             disabled:opacity-15 disabled:cursor-not-allowed
                             bg-gradient-to-br from-emerald-500 to-teal-600
                             hover:from-emerald-400 hover:to-teal-500
                             text-white shadow-[0_0_20px_-4px_rgba(16,185,129,0.3)]
                             disabled:shadow-none disabled:from-zinc-700 disabled:to-zinc-800"
                  title="Send message"
                >
                  <Send size={14} />
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Footer info */}
        <div className="flex items-center justify-center gap-1.5 mt-3">
          <Sparkles size={10} className="text-zinc-700" />
          <p className="text-2xs text-zinc-600 tracking-wide">
            Layered Memory Architecture — behavior-adaptive retrieval with structured cognition
          </p>
        </div>
      </div>
    </div>
  );
}
