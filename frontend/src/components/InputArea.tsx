import { useState, useRef, useEffect } from 'react';
import { Send, Square } from 'lucide-react';
import type { useChatStream } from '../hooks/use-chat-stream';

interface Props {
  chat: ReturnType<typeof useChatStream>;
}

export default function InputArea({ chat }: Props) {
  const [value, setValue] = useState('');
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
    <div className="border-t border-sidebar-border/50 bg-chat-bg">
      <div className="max-w-3xl mx-auto px-4 py-3">
        <div className="flex items-end gap-2 bg-input-bg rounded-2xl border border-input-border
                        focus-within:border-accent/50 transition-colors px-4 py-3">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message RAG Chat…"
            rows={1}
            className="flex-1 bg-transparent outline-none resize-none text-sm text-gray-200
                       placeholder:text-sidebar-muted max-h-[200px]"
          />

          {chat.isLoading ? (
            <button
              onClick={chat.stop}
              className="flex-shrink-0 p-2 rounded-lg bg-white/10 text-white
                         hover:bg-white/20 transition-colors"
              title="Stop generating"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!value.trim()}
              className="flex-shrink-0 p-2 rounded-lg bg-accent text-white
                         hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed
                         transition-colors"
              title="Send message"
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <p className="text-[10px] text-sidebar-muted text-center mt-2">
          RAG Chat uses retrieval-augmented generation. Responses may not always be accurate.
        </p>
      </div>
    </div>
  );
}
