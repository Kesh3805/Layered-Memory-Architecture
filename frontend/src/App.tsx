import { useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useChatStore } from './store';
import { useChatStream } from './hooks/use-chat-stream';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ProfileModal from './components/ProfileModal';
import CommandPalette from './components/CommandPalette';
import { AIResearchDashboard } from './components/ai';

export default function App() {
  const {
    sidebarOpen,
    profileModalOpen,
    refreshConversations,
    refreshProfile,
    commandPaletteOpen,
    setCommandPaletteOpen,
  } = useChatStore();
  const chat = useChatStream();

  // Bootstrap: load conversations + profile on mount
  useEffect(() => {
    refreshConversations();
    refreshProfile();
  }, [refreshConversations, refreshProfile]);

  // Global keyboard shortcut: Ctrl+K / Cmd+K for command palette
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(!commandPaletteOpen);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [commandPaletteOpen, setCommandPaletteOpen]);

  return (
    <div className="flex h-screen overflow-hidden bg-chat-bg relative noise-overlay">
      {/* Ambient glow */}
      <div className="ambient-glow" />

      {/* Sidebar */}
      <motion.div
        animate={{ width: sidebarOpen ? 272 : 0 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="flex-shrink-0 overflow-hidden relative z-10"
      >
        <Sidebar />
      </motion.div>

      {/* Main chat area */}
      <ChatArea chat={chat} />

      {/* Profile modal */}
      <AnimatePresence>
        {profileModalOpen && <ProfileModal />}
      </AnimatePresence>

      {/* Research dashboard overlay */}
      <AIResearchDashboard />

      {/* Command palette */}
      <CommandPalette />
    </div>
  );
}
