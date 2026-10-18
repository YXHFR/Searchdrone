import { create } from "zustand";
import type { ChatMessage } from "../api/types";

export interface PanelChatMessage {
  /** ChatMessage.id, or the client_msg_id while it is still being sent. */
  key: string;
  clientMsgId: string | null;
  seq: number | null;
  role: ChatMessage["role"];
  kind: ChatMessage["kind"];
  text: string;
  createdAt: string;
  status: "sending" | "sent" | "failed";
  inReplyTo: string | null;
  /** refs.proposed_action, e.g. "resume" on an AI proposal (draws the Resume button). */
  proposedAction: string | null;
}

interface ChatState {
  messages: PanelChatMessage[];
  /** `next_before` from the last history page; null = nothing older. */
  nextBefore: number | null;
  addLocal(m: PanelChatMessage): void;
  markFailed(clientMsgId: string): void;
  /** Server message; replaces the optimistic copy with the same client_msg_id. */
  receive(m: ChatMessage): void;
  /** A history page (any order); merged without duplicates. */
  receivePage(items: ChatMessage[], nextBefore?: number | null): void;
  clear(): void;
}

/** Chat order: the server's time, then the id for messages with the same time. */
export function chatOrder(
  a: { createdAt: string; key: string },
  b: { createdAt: string; key: string },
): number {
  return a.createdAt.localeCompare(b.createdAt) || a.key.localeCompare(b.key);
}

function toPanel(m: ChatMessage): PanelChatMessage {
  const clientMsgId = typeof m.refs.client_msg_id === "string" ? m.refs.client_msg_id : null;
  const proposed = m.refs.proposed_action;
  return {
    key: m.id,
    clientMsgId,
    seq: m.seq,
    role: m.role,
    kind: m.kind,
    text: m.text,
    createdAt: m.created_at,
    status: "sent",
    inReplyTo: m.in_reply_to,
    proposedAction: typeof proposed === "string" ? proposed : null,
  };
}

function merge(list: PanelChatMessage[], incoming: PanelChatMessage[]): PanelChatMessage[] {
  let out = list;
  for (const msg of incoming) {
    out = out.filter(
      (x) => x.key !== msg.key && (msg.clientMsgId === null || x.clientMsgId !== msg.clientMsgId),
    );
    out = [...out, msg];
  }
  return [...out].sort(chatOrder);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  nextBefore: null,
  addLocal: (m) => set((s) => ({ messages: [...s.messages, m].sort(chatOrder) })),
  markFailed: (id) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.clientMsgId === id ? { ...m, status: "failed" } : m)),
    })),
  receive: (m) => set((s) => ({ messages: merge(s.messages, [toPanel(m)]) })),
  receivePage: (items, nextBefore) =>
    set((s) => ({
      messages: merge(s.messages, items.map(toPanel)),
      nextBefore: nextBefore === undefined ? s.nextBefore : nextBefore,
    })),
  clear: () => set({ messages: [], nextBefore: null }),
}));

/** Highest server seq seen, for `after_seq` after a reconnect. */
export function lastChatSeq(messages: PanelChatMessage[]): number | null {
  let max: number | null = null;
  for (const m of messages) if (m.seq !== null && (max === null || m.seq > max)) max = m.seq;
  return max;
}
