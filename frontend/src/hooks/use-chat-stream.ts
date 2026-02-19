/**
 * Custom streaming chat hook — wraps Vercel AI SDK `useChat` with our
 * backend's request/response format.
 *
 * Our backend expects:  { user_query, conversation_id }
 * AI SDK sends:         { messages, ...body }
 *
 * We override `fetch` to transform the request while keeping all the
 * nice streaming UX (auto message management, loading state, stop).
 */

import { useChat, type Message } from 'ai/react';
import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../store';
import * as api from '../api';

export function useChatStream() {
  const {
    conversationId,
    setConversationId,
    addConversation,
    refreshConversations,
  } = useChatStore();

  // Keep conversationId in a ref so the fetch closure always has the latest
  const cidRef = useRef(conversationId);
  useEffect(() => { cidRef.current = conversationId; }, [conversationId]);

  const chat = useChat({
    api: '/chat/stream',
    streamProtocol: 'data',

    // Transform AI SDK request → our backend format
    fetch: async (url, init) => {
      const body = JSON.parse((init?.body as string) || '{}');
      const msgs: Message[] = body.messages || [];
      const lastMsg = msgs[msgs.length - 1];

      return fetch(url as string, {
        ...init,
        body: JSON.stringify({
          user_query: lastMsg?.content || '',
          conversation_id: cidRef.current,
        }),
      });
    },

    onFinish: () => {
      // Refresh conversations after 1.5s to pick up auto-generated title
      setTimeout(() => refreshConversations(), 1500);
    },

    onError: (err) => {
      console.error('Chat stream error:', err);
    },
  });

  // ── Load messages when switching conversations ────────────────────────
  useEffect(() => {
    if (conversationId) {
      api.getMessages(conversationId).then(({ messages }) => {
        const mapped: Message[] = messages.map((m, i) => ({
          id: String(m.id ?? `loaded-${i}`),
          role: m.role as 'user' | 'assistant',
          content: m.content,
        }));
        chat.setMessages(mapped);
      }).catch(console.error);
    } else {
      chat.setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  // ── Send helper: creates conversation on first message ────────────────
  const send = useCallback(async (content: string) => {
    let cid = cidRef.current;
    if (!cid) {
      const conv = await api.createConversation('New Chat');
      cid = conv.id;
      cidRef.current = cid;
      setConversationId(cid);
      addConversation(conv);
    }
    chat.append({ role: 'user', content });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setConversationId, addConversation]);

  return {
    messages: chat.messages,
    isLoading: chat.isLoading,
    stop: chat.stop,
    send,
    input: chat.input,
    setInput: chat.setInput,
    setMessages: chat.setMessages,
  };
}
