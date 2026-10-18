import { X } from "lucide-react";
import { type KeyboardEvent, useEffect, useId, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { type PanelApproval, toPanelApproval } from "../../api/approvals";
import { api } from "../../api/http";
import { fetchSubject, loadOlderChat } from "../../api/live";
import { paths } from "../../api/paths";
import type { ChatMessage } from "../../api/types";
import { useNow } from "../../lib/useNow";
import { uuidv7 } from "../../lib/uuidv7";
import { ClearanceControl } from "../../safety/ClearanceControl";
import { ResumeButton, ResumeRow } from "../../safety/ResumeButton";
import { resumeView } from "../../safety/resume";
import { useBrainState, useCommandMode } from "../../state/brain";
import { chatOrder, type PanelChatMessage, useChatStore } from "../../state/chatStore";
import { useLiveStore } from "../../state/liveStore";
import { useMissionStore } from "../../state/missionStore";
import { useIsPilot } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";
import type { ViewProps } from "../registry";
import { ApprovalCard } from "./cards/ApprovalCard";
import { OrderButtons } from "./OrderButtons";

/** The server's limit for one chat message (ChatPost.text). */
export const CHAT_MAX_CHARS = 4000;
const COUNTER_FROM = 3800;

/** The fixed commands the command mode understands, offered as buttons while the AI does
 * not answer. Each sends its text as an ordinary chat line. */
const COMMAND_BUTTONS = [
  { key: "status", text: "status" },
  { key: "hold", text: "håll position" },
  { key: "resume", text: "fortsätt" },
  { key: "help", text: "hjälp" },
] as const;

/** Post one operator chat line (shown at once as "sending", marked failed on error). */
async function postChat(missionId: string, body: string, replyTo: string | null): Promise<void> {
  const clientMsgId = uuidv7();
  const chat = useChatStore.getState();
  chat.addLocal({
    key: clientMsgId,
    clientMsgId,
    seq: null,
    role: "operator",
    kind: "text",
    text: body,
    createdAt: new Date().toISOString(),
    status: "sending",
    inReplyTo: replyTo,
    proposedAction: null,
  });
  try {
    const saved = await api<ChatMessage>(paths.missionChat(missionId), {
      method: "POST",
      body: {
        client_msg_id: clientMsgId,
        text: body,
        annotation_ids: [],
        ...(replyTo ? { in_reply_to: replyTo } : {}),
      },
    });
    chat.receive({ ...saved, refs: { ...saved.refs, client_msg_id: clientMsgId } });
  } catch {
    chat.markFailed(clientMsgId);
  }
}

type Item =
  | { type: "msg"; key: string; createdAt: string; msg: PanelChatMessage }
  | { type: "card"; key: string; createdAt: string; card: PanelApproval };

/** Messages and approval cards, one list sorted on server time. */
function useItems(): Item[] {
  const messages = useChatStore((s) => s.messages);
  const approvals = useMissionStore((s) => s.approvals);
  const charters = useMissionStore((s) => s.charters);
  const commands = useMissionStore((s) => s.commands);
  return useMemo(() => {
    const cards: Item[] = Object.values(approvals).flatMap((a) => {
      const card = toPanelApproval(a, { charters, commands });
      return card
        ? [{ type: "card" as const, key: card.approval_id, createdAt: card.created_at, card }]
        : [];
    });
    const msgs: Item[] = messages.map((m) => ({
      type: "msg",
      key: m.key,
      createdAt: m.createdAt,
      msg: m,
    }));
    return [...msgs, ...cards].sort(chatOrder);
  }, [messages, approvals, charters, commands]);
}

function Message({ m, vehicleId }: { m: PanelChatMessage; vehicleId: string }) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const vehicle = useLiveStore((s) => s.vehicle);
  const time = new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" });
  const question = m.role === "brain" && m.kind === "question";
  const proposeResume = m.kind === "proposal" && m.proposedAction === "resume";
  return (
    <li
      className={`msg msg-${m.role}${question ? " msg-question" : ""}`}
      data-status={m.status}
      data-kind={m.kind}
      data-testid={question ? "question" : undefined}
    >
      <small>
        {question ? `${t("chat.question")} · ` : null}
        {t(`chat.role.${m.role}`)} · {time.format(new Date(m.createdAt))}
        {m.status === "sending" ? ` · ${t("chat.sending")}` : null}
        {m.status === "failed" ? ` · ${t("chat.failed")}` : null}
      </small>
      {/* Server and AI text is always rendered as a text node, never as HTML. */}
      <span className="msg-text">{m.text}</span>
      {question && isPilot ? (
        <button
          type="button"
          className="btn btn-small btn-outline answer-btn"
          data-testid="answer"
          onClick={() =>
            useUiStore
              .getState()
              .setDraft({ replyTo: { id: m.key, preview: m.text.slice(0, 60) } }, true)
          }
        >
          {t("chat.answer")}
        </button>
      ) : null}
      {proposeResume && isPilot ? (
        resumeView(vehicle) === "button" ? (
          <ResumeButton vehicleId={vehicleId} />
        ) : (
          <span className="muted">{t("resume.nothing")}</span>
        )
      ) : null}
    </li>
  );
}

function MessageList({ vehicleId }: ViewProps) {
  const { t } = useTranslation();
  const items = useItems();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const nextBefore = useChatStore((s) => s.nextBefore);
  const approvals = useMissionStore((s) => s.approvals);
  const charters = useMissionStore((s) => s.charters);
  const commands = useMissionStore((s) => s.commands);
  const end = useRef<HTMLLIElement>(null);
  const count = items.length;
  useEffect(() => {
    if (count > 0) end.current?.scrollIntoView?.({ block: "end" });
  }, [count]);
  // A card whose subject has not arrived yet asks for it once.
  useEffect(() => {
    for (const a of Object.values(approvals)) {
      if (a.status !== "pending") continue;
      const have =
        a.kind === "charter"
          ? charters[a.subject_id]
          : a.kind === "command"
            ? commands[a.subject_id]
            : true;
      if (!have) void fetchSubject(a);
    }
  }, [approvals, charters, commands]);
  if (items.length === 0) return <p className="muted chat-empty">{t("chat.empty")}</p>;
  return (
    <ol className="msgs" aria-live="polite" data-testid="messages">
      {nextBefore !== null && missionId ? (
        <li className="msgs-older">
          <button
            type="button"
            className="btn btn-small btn-outline"
            onClick={() => void loadOlderChat(missionId)}
          >
            {t("chat.older")}
          </button>
        </li>
      ) : null}
      {items.map((it) =>
        it.type === "msg" ? (
          <Message key={it.key} m={it.msg} vehicleId={vehicleId} />
        ) : (
          <li key={it.key} className="msg-card">
            <ApprovalCard approval={it.card} />
          </li>
        ),
      )}
      <li ref={end} aria-hidden="true" className="msgs-end" />
    </ol>
  );
}

/** "AI:n tänker… N s", or the command-mode / AI-down notices (one line above the composer).
 * In the command mode the fixed commands are offered as buttons below the line. */
export function BrainStatusRow() {
  const { t } = useTranslation();
  const since = useLiveStore((s) => s.brainBusySince);
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const isPilot = useIsPilot();
  const brain = useBrainState();
  const now = useNow(500);
  const command = useCommandMode();
  let text: string | null = null;
  let warn = false;
  if (brain === "down") {
    text = t("chat.brainDown");
    warn = true;
  } else if (command) {
    text = t("chat.brainDegraded");
    warn = true;
  } else if (since !== null) {
    text = t("chat.thinking", { s: Math.max(0, Math.floor((now - since) / 1000)) });
  }
  return (
    <>
      <p
        className={warn ? "brain-row brain-row-warn" : "brain-row"}
        aria-live="polite"
        data-testid="brain-row"
      >
        {text ?? ""}
      </p>
      {command && isPilot && missionId !== null ? (
        <fieldset className="orders" data-testid="command-buttons">
          <legend className="sr-only">{t("chat.commandButtons")}</legend>
          {COMMAND_BUTTONS.map((b) => (
            <button
              key={b.key}
              type="button"
              className="btn btn-small btn-outline"
              onClick={() => void postChat(missionId, b.text, null)}
            >
              {t(`chat.command.${b.key}`)}
            </button>
          ))}
        </fieldset>
      ) : null}
    </>
  );
}

function Composer() {
  const { t } = useTranslation();
  const id = useId();
  const ref = useRef<HTMLTextAreaElement>(null);
  const isPilot = useIsPilot();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const draft = useUiStore((s) => s.draft);
  const text = draft.text;
  const tooLong = text.length > CHAT_MAX_CHARS;
  const canSend = isPilot && missionId !== null && text.trim().length > 0 && !tooLong;
  useEffect(() => {
    if (draft.focusTick > 0) ref.current?.focus();
  }, [draft.focusTick]);

  async function send(): Promise<void> {
    if (!canSend || missionId === null) return;
    const body = text;
    const replyTo = draft.replyTo?.id ?? null;
    useUiStore.getState().setDraft({ text: "", replyTo: null });
    await postChat(missionId, body, replyTo);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  const hint = !isPilot ? t("chat.observer") : missionId === null ? t("chat.noMission") : null;
  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        void send();
      }}
    >
      {draft.replyTo ? (
        <p className="reply-to" data-testid="reply-to">
          {t("chat.replyTo", { preview: draft.replyTo.preview })}
          <button
            type="button"
            className="icon-btn"
            aria-label={t("chat.replyCancel")}
            onClick={() => useUiStore.getState().setDraft({ replyTo: null })}
          >
            <X aria-hidden="true" size={16} />
          </button>
        </p>
      ) : null}
      <label htmlFor={id} className="sr-only">
        {t("chat.inputLabel")}
      </label>
      <textarea
        id={id}
        ref={ref}
        className="input"
        rows={2}
        value={text}
        disabled={!isPilot}
        placeholder={t("chat.placeholder")}
        onChange={(e) => useUiStore.getState().setDraft({ text: e.target.value })}
        onKeyDown={onKey}
      />
      <button type="submit" className="btn btn-primary" disabled={!canSend}>
        {t("chat.send")}
      </button>
      {text.length >= COUNTER_FROM ? (
        <span className={tooLong ? "error" : "muted"}>
          {t("chat.counter", { n: text.length, max: CHAT_MAX_CHARS })}
          {tooLong ? ` · ${t("chat.tooLong")}` : ""}
        </span>
      ) : null}
      {hint ? <p className="muted composer-hint">{hint}</p> : null}
    </form>
  );
}

export function ChatView(props: ViewProps) {
  return (
    <div className="view-body chat" id="chat">
      <MessageList {...props} />
      <ResumeRow vehicleId={props.vehicleId} />
      <ClearanceControl />
      <OrderButtons />
      <BrainStatusRow />
      <Composer />
    </div>
  );
}
