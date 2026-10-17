import { beforeEach, describe, expect, it } from "vitest";
import type { ChatMessage } from "../api/types";
import { lastChatSeq, useChatStore } from "./chatStore";

const msg = (p: Partial<ChatMessage>): ChatMessage =>
  ({
    id: "m1",
    mission_id: "M1",
    seq: 1,
    role: "brain",
    kind: "text",
    text: "x",
    created_at: "2026-10-16T10:00:00Z",
    in_reply_to: null,
    refs: {},
    ...p,
  }) as ChatMessage;

beforeEach(() => useChatStore.getState().clear());

describe("chatStore", () => {
  it("the echo with the same client_msg_id replaces the optimistic message", () => {
    const s = useChatStore.getState();
    s.addLocal({
      key: "c1",
      clientMsgId: "c1",
      seq: null,
      role: "operator",
      kind: "text",
      text: "hej",
      createdAt: "2026-10-16T10:00:05Z",
      status: "sending",
      inReplyTo: null,
      proposedAction: null,
    });
    s.receive(
      msg({
        id: "s1",
        seq: 2,
        role: "operator",
        text: "hej",
        created_at: "2026-10-16T10:00:06Z",
        refs: { client_msg_id: "c1" },
      }),
    );
    s.receive(
      msg({
        id: "s1",
        seq: 2,
        role: "operator",
        text: "hej",
        created_at: "2026-10-16T10:00:06Z",
        refs: { client_msg_id: "c1" },
      }),
    );
    expect(useChatStore.getState().messages.map((m) => [m.key, m.status])).toEqual([
      ["s1", "sent"],
    ]);
  });

  it("sorts on server time, then id; pages merge without duplicates", () => {
    const s = useChatStore.getState();
    s.receivePage(
      [
        msg({ id: "b", seq: 3, created_at: "2026-10-16T10:00:03Z" }),
        msg({ id: "a", seq: 2, created_at: "2026-10-16T10:00:03Z" }),
      ],
      2,
    );
    s.receivePage(
      [
        msg({ id: "z", seq: 1, created_at: "2026-10-16T10:00:01Z" }),
        msg({ id: "a", seq: 2, created_at: "2026-10-16T10:00:03Z" }),
      ],
      null,
    );
    expect(useChatStore.getState().messages.map((m) => m.key)).toEqual(["z", "a", "b"]);
    expect(useChatStore.getState().nextBefore).toBeNull();
    expect(lastChatSeq(useChatStore.getState().messages)).toBe(3);
  });

  it("keeps the resume proposal and the reply target", () => {
    useChatStore
      .getState()
      .receive(
        msg({ id: "p", kind: "proposal", refs: { proposed_action: "resume" }, in_reply_to: "q" }),
      );
    expect(useChatStore.getState().messages[0]).toMatchObject({
      proposedAction: "resume",
      inReplyTo: "q",
    });
  });
});
