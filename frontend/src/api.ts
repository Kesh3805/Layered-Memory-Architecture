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
