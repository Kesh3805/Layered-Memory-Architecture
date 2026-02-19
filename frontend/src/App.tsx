import { useEffect } from 'react';
import { useChatStore } from './store';
import { useChatStream } from './hooks/use-chat-stream';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ProfileModal from './components/ProfileModal';
import CommandPalette from './components/CommandPalette';

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
    <div className="flex h-screen overflow-hidden bg-chat-bg">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? 'w-64' : 'w-0'
        } transition-all duration-200 flex-shrink-0 overflow-hidden`}
      >
        <Sidebar />
      </div>

      {/* Main chat area */}
      <ChatArea chat={chat} />

      {/* Profile modal */}
      {profileModalOpen && <ProfileModal />}

      {/* Command palette */}
      <CommandPalette />
    </div>
  );
}
