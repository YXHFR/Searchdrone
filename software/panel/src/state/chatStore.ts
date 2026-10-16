import { create } from "zustand";
import type { ChatMessage } from "../api/types";

export interface PanelChatMessage {
  /** ChatMessage.id, or the client_msg_id while it is still being sent. */
  key: string;
  clientMsgId: string | null;
  role: ChatMessage["role"];
  kind: ChatMessage["kind"];
  text: string;
  createdAt: string;
  status: "sending" | "sent" | "failed";
}

interface ChatState {
  messages: PanelChatMessage[];
  addLocal(m: PanelChatMessage): void;
  markFailed(clientMsgId: string): void;
  /** Server message; replaces the optimistic copy with the same client_msg_id. */
  receive(m: ChatMessage): void;
  clear(): void;
}

function sorted(list: PanelChatMessage[]): PanelChatMessage[] {
  return [...list].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  addLocal: (m) => set((s) => ({ messages: sorted([...s.messages, m]) })),
  markFailed: (id) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.clientMsgId === id ? { ...m, status: "failed" } : m)),
    })),
  receive: (m) =>
    set((s) => {
      const clientMsgId = typeof m.refs.client_msg_id === "string" ? m.refs.client_msg_id : null;
      const rest = s.messages.filter(
        (x) => x.key !== m.id && (clientMsgId === null || x.clientMsgId !== clientMsgId),
      );
      const msg: PanelChatMessage = {
        key: m.id,
        clientMsgId,
        role: m.role,
        kind: m.kind,
        text: m.text,
        createdAt: m.created_at,
        status: "sent",
      };
      return { messages: sorted([...rest, msg]) };
    }),
  clear: () => set({ messages: [] }),
}));
