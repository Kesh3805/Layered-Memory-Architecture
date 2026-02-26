/* ── API client — typed fetch wrappers for every backend endpoint ─────── */

const BASE = '';  // same origin in production; Vite proxy in dev

export interface Conversation {
  id: string;
  title: string;
  created_at: string | null;
  updated_at: string | null;
  message_count: number;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ProfileEntry {
  id: number;
  key: string;
  value: string;
  category: string;
}

/* ── Conversations ──────────────────────────────────────────────────────── */

export async function listConversations(limit = 50): Promise<{ conversations: Conversation[]; count: number }> {
  const res = await fetch(`${BASE}/conversations?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to list conversations');
  return res.json();
}

export async function createConversation(title = 'New Chat'): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error('Failed to create conversation');
  return res.json();
}

export async function renameConversation(id: string, title: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error('Failed to rename conversation');
  return res.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`${BASE}/conversations/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete conversation');
}

export async function getMessages(conversationId: string, limit = 200): Promise<{ messages: ChatMessage[] }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to get messages');
  return res.json();
}

/* ── Profile ────────────────────────────────────────────────────────────── */

export async function getProfile(): Promise<{ entries: ProfileEntry[]; count: number }> {
  const res = await fetch(`${BASE}/profile`);
  if (!res.ok) throw new Error('Failed to get profile');
  return res.json();
}

export async function addProfileEntry(key: string, value: string, category = 'general'): Promise<ProfileEntry> {
  const res = await fetch(`${BASE}/profile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, category }),
  });
  if (!res.ok) throw new Error('Failed to add profile entry');
  return res.json();
}

export async function deleteProfileEntry(id: number): Promise<void> {
  const res = await fetch(`${BASE}/profile/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete profile entry');
}

/* ── Threads ────────────────────────────────────────────────────────────── */

export interface Thread {
  thread_id: string;
  conversation_id: string;
  centroid: number[] | null;
  message_count: number;
  created_at: string | null;
  updated_at: string | null;
  summary: string | null;
  label: string | null;
}

export async function getThreads(conversationId: string): Promise<{ threads: Thread[] }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/threads`);
  if (!res.ok) throw new Error('Failed to get threads');
  return res.json();
}

export async function getThread(conversationId: string, threadId: string): Promise<Thread> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/threads/${threadId}`);
  if (!res.ok) throw new Error('Failed to get thread');
  return res.json();
}

/* ── Research Insights ──────────────────────────────────────────────────── */

export interface Insight {
  id: number;
  conversation_id: string;
  thread_id: string | null;
  insight_type: string;
  content: string;
  confidence: number;
  created_at: string | null;
}

export interface ConceptLink {
  id: number;
  conversation_id: string;
  concept: string;
  thread_id: string | null;
  created_at: string | null;
}

export async function getInsights(conversationId: string): Promise<{ insights: Insight[] }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/insights`);
  if (!res.ok) throw new Error('Failed to get insights');
  return res.json();
}

export async function getConcepts(conversationId: string): Promise<{ concepts: ConceptLink[] }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/concepts`);
  if (!res.ok) throw new Error('Failed to get concepts');
  return res.json();
}

export async function searchInsights(
  q: string,
  opts?: { k?: number; type?: string; conversation_id?: string },
): Promise<{ results: Insight[] }> {
  const params = new URLSearchParams({ q });
  if (opts?.k) params.set('k', String(opts.k));
  if (opts?.type) params.set('type', opts.type);
  if (opts?.conversation_id) params.set('conversation_id', opts.conversation_id);
  const res = await fetch(`${BASE}/insights/search?${params}`);
  if (!res.ok) throw new Error('Failed to search insights');
  return res.json();
}

export async function searchConcepts(q: string): Promise<{ concepts: ConceptLink[] }> {
  const res = await fetch(`${BASE}/concepts/search?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error('Failed to search concepts');
  return res.json();
}
