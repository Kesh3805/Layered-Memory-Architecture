import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Square } from 'lucide-react';
import type { useChatStream } from '../hooks/use-chat-stream';

interface Props {
  chat: ReturnType<typeof useChatStream>;
}

export default function InputArea({ chat }: Props) {
  const [value, setValue] = useState('');
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
    <div className="border-t border-surface-2/20 bg-gradient-to-t from-chat-bg via-chat-bg to-transparent">
      <div className="max-w-3xl mx-auto px-4 py-4">
        <div
          className={`relative flex items-end gap-2 rounded-2xl border px-4 py-3
                     bg-surface-1/50 transition-all duration-300
                     ${focused
                       ? 'border-accent/30 shadow-glow-sm'
                       : 'border-surface-2/50 hover:border-surface-3/50'}`}
        >
          {/* Animated glow ring when focused */}
          {focused && (
            <div className="absolute inset-0 rounded-2xl ring-1 ring-accent/10 pointer-events-none" />
          )}

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="Message RAG Chat…"
            rows={1}
            className="flex-1 bg-transparent outline-none resize-none text-sm text-zinc-200
                       placeholder:text-zinc-600 max-h-[200px] leading-relaxed"
          />

          <AnimatePresence mode="wait">
            {chat.isLoading ? (
              <motion.button
                key="stop"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ duration: 0.15 }}
                onClick={chat.stop}
                className="flex-shrink-0 p-2 rounded-xl bg-zinc-700/50 text-zinc-300
                           hover:bg-zinc-600/50 transition-colors"
                title="Stop generating"
              >
                <Square size={15} />
              </motion.button>
            ) : (
              <motion.button
                key="send"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ duration: 0.15 }}
                onClick={handleSubmit}
                disabled={!value.trim()}
                className="flex-shrink-0 p-2 rounded-xl bg-accent text-white
                           hover:bg-accent-hover disabled:opacity-20 disabled:cursor-not-allowed
                           transition-all duration-200 shadow-glow-sm disabled:shadow-none"
                title="Send message"
              >
                <Send size={15} />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
        <p className="text-2xs text-zinc-600 text-center mt-2.5 tracking-wide">
          RAG Chat uses retrieval-augmented generation. Responses may not always be accurate.
        </p>
      </div>
    </div>
  );
}
