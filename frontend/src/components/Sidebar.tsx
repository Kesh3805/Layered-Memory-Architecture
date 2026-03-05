import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  User,
  Pencil,
  Check,
  X,
  Bug,
  Command,
  Brain,
  Shield,
  Database,
  Code2,
  BookOpen,
  Globe,
  HelpCircle,
} from 'lucide-react';
import { useChatStore } from '../store';

/* ── Category icon detection from conversation title ───────────────────── */

const categoryRules: { test: RegExp; icon: React.ReactNode; color: string }[] = [
  { test: /\b(privacy|private|secure|security|delete|forget)\b/i, icon: <Shield size={12} />, color: 'text-red-400' },
  { test: /\b(machine learning|ML|neural|model|train|AI|LLM|GPT)\b/i, icon: <Brain size={12} />, color: 'text-purple-400' },
  { test: /\b(database|SQL|postgres|pgvector|query|table)\b/i, icon: <Database size={12} />, color: 'text-cyan-400' },
  { test: /\b(code|python|javascript|typescript|react|function|class|API)\b/i, icon: <Code2 size={12} />, color: 'text-emerald-400' },
  { test: /\b(RAG|retrieval|embedding|FAISS|vector|chunk)\b/i, icon: <BookOpen size={12} />, color: 'text-amber-400' },
  { test: /\b(web|HTTP|URL|fetch|deploy|docker)\b/i, icon: <Globe size={12} />, color: 'text-blue-400' },
  { test: /\b(help|how|what|why|explain)\b/i, icon: <HelpCircle size={12} />, color: 'text-zinc-400' },
];

function getCategoryIcon(title: string): { icon: React.ReactNode; color: string } {
  for (const rule of categoryRules) {
    if (rule.test.test(title)) return { icon: rule.icon, color: rule.color };
  }
  return { icon: <MessageSquare size={12} />, color: 'text-sidebar-muted' };
}

export default function Sidebar() {
  const {
    conversations,
    conversationId,
    setConversationId,
    removeConversation,
    renameConversation,
    toggleSidebar,
    setProfileModalOpen,
    debugMode,
    toggleDebugMode,
    setCommandPaletteOpen,
    toggleResearchDash,
  } = useChatStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const handleNew = () => setConversationId(null);

  const startEdit = (id: string, title: string) => {
    setEditingId(id);
    setEditTitle(title);
  };

  const confirmEdit = async () => {
    if (editingId && editTitle.trim()) {
      await renameConversation(editingId, editTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <div className="flex flex-col h-full bg-sidebar-bg border-r border-surface-2/50 w-[272px]">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-surface-2/50">
        <button
          onClick={handleNew}
          className="flex items-center gap-2.5 px-3 py-2.5 text-sm text-sidebar-text rounded-xl
                     hover:bg-surface-1 transition-all duration-200 flex-1 group"
        >
          <span className="p-1 rounded-lg bg-accent/10 text-accent group-hover:bg-accent/20 transition-colors">
            <Plus size={14} />
          </span>
          <span className="font-medium">New Chat</span>
        </button>
        <button
          onClick={toggleSidebar}
          title="Close sidebar"
          className="p-2 text-sidebar-muted hover:text-sidebar-text rounded-xl
                     hover:bg-surface-1 transition-all duration-200"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2 px-1.5">
        {conversations.length === 0 && (
          <p className="text-sidebar-muted text-xs text-center py-8 opacity-60">
            No conversations yet
          </p>
        )}
        <AnimatePresence initial={false}>
          {conversations.map((c) => (
            <motion.div
              key={c.id}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            >
              <div
                className={`group flex items-center gap-2.5 mx-1 px-3 py-2.5 rounded-xl cursor-pointer
                  transition-all duration-200 text-sm mb-0.5 ${
                    c.id === conversationId
                      ? 'bg-surface-1 text-white shadow-inner-glow border border-surface-2/50'
                      : 'text-sidebar-text hover:bg-surface-1/50'
                  }`}
                onClick={() => {
                  if (editingId !== c.id) setConversationId(c.id);
                }}
              >
                {(() => {
                  const cat = getCategoryIcon(c.title);
                  return <span className={`flex-shrink-0 opacity-70 ${cat.color}`}>{cat.icon}</span>;
                })()}

                {editingId === c.id ? (
                  <div className="flex items-center gap-1 flex-1 min-w-0">
                    <input
                      title="Rename conversation"
                      placeholder="Conversation name"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') confirmEdit();
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      className="flex-1 bg-transparent border-b border-accent outline-none text-sm min-w-0"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                    <button onClick={(e) => { e.stopPropagation(); confirmEdit(); }}
                      title="Confirm" className="p-0.5 hover:text-accent"><Check size={12} /></button>
                    <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }}
                      title="Cancel" className="p-0.5 hover:text-danger"><X size={12} /></button>
                  </div>
                ) : (
                  <>
                    <span className="truncate flex-1 text-[13px]">{c.title}</span>
                    <div className="hidden group-hover:flex items-center gap-0.5">
                      <button
                        title="Rename"
                        onClick={(e) => { e.stopPropagation(); startEdit(c.id, c.title); }}
                        className="p-1 rounded-lg hover:bg-surface-2/50 text-sidebar-muted hover:text-sidebar-text transition-colors"
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        title="Delete"
                        onClick={(e) => { e.stopPropagation(); removeConversation(c.id); }}
                        className="p-1 rounded-lg hover:bg-danger-muted text-sidebar-muted hover:text-danger transition-colors"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="border-t border-surface-2/50 p-2.5 space-y-0.5">
        <SidebarButton icon={<User size={15} />} label="Profile & Memory" onClick={() => setProfileModalOpen(true)} />
        <SidebarButton icon={<Brain size={15} />} label="Research Dashboard" onClick={toggleResearchDash} />
        <SidebarButton
          icon={<Bug size={15} />}
          label="Debug Mode"
          onClick={toggleDebugMode}
          active={debugMode}
          activeColor="text-yellow-400"
          badge={debugMode ? 'ON' : undefined}
        />
        <SidebarButton
          icon={<Command size={15} />}
          label="Commands"
          onClick={() => setCommandPaletteOpen(true)}
          shortcut="Ctrl+K"
        />
      </div>
    </div>
  );
}

/* ── Sidebar button ────────────────────────────────────────────────────── */

function SidebarButton({
  icon, label, onClick, active, activeColor, badge, shortcut,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  active?: boolean;
  activeColor?: string;
  badge?: string;
  shortcut?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2.5 w-full px-3 py-2 text-[13px] rounded-xl
                 transition-all duration-200 hover:bg-surface-1/50
                 ${active ? activeColor || 'text-accent' : 'text-sidebar-muted hover:text-sidebar-text'}`}
    >
      <span className="opacity-80">{icon}</span>
      <span className="flex-1 text-left">{label}</span>
      {badge && (
        <span className="text-2xs bg-yellow-500/20 text-yellow-400 px-1.5 py-0.5 rounded-full font-medium">
          {badge}
        </span>
      )}
      {shortcut && (
        <span className="text-2xs text-sidebar-muted/50 font-mono">{shortcut}</span>
      )}
    </button>
  );
}
