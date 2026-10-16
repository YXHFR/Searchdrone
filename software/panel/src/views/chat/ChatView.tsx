import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../../api/http";
import { paths } from "../../api/paths";
import type { ChatMessage } from "../../api/types";
import { uuidv7 } from "../../lib/uuidv7";
import { ResumeRow } from "../../safety/ResumeButton";
import { useChatStore } from "../../state/chatStore";
import { useLiveStore } from "../../state/liveStore";
import { useIsPilot } from "../../state/sessionStore";
import type { ViewProps } from "../registry";

export const CHAT_MAX_CHARS = 4000;
const COUNTER_FROM = 3800;

/** Fallback orders that go through the safety gate like the AI's. Wired in a later step. */
const ORDER_BUTTONS = ["orders.gotoPin", "orders.survey", "orders.land"] as const;

function MessageList() {
  const { t } = useTranslation();
  const messages = useChatStore((s) => s.messages);
  const end = useRef<HTMLLIElement>(null);
  const count = messages.length;
  useEffect(() => {
    if (count > 0) end.current?.scrollIntoView?.({ block: "end" });
  }, [count]);
  const time = new Intl.DateTimeFormat("sv-SE", { hour: "2-digit", minute: "2-digit" });
  if (messages.length === 0) return <p className="muted chat-empty">{t("chat.empty")}</p>;
  return (
    <ol className="msgs" aria-live="polite">
      {messages.map((m) => (
        <li key={m.key} className={`msg msg-${m.role}`} data-status={m.status}>
          <small>
            {t(`chat.role.${m.role}`)} · {time.format(new Date(m.createdAt))}
            {m.status === "sending" ? ` · ${t("chat.sending")}` : null}
            {m.status === "failed" ? ` · ${t("chat.failed")}` : null}
          </small>
          {/* Server and AI text is always rendered as a text node, never as HTML. */}
          <span className="msg-text">{m.text}</span>
        </li>
      ))}
      <li ref={end} aria-hidden="true" className="msgs-end" />
    </ol>
  );
}

function Composer() {
  const { t } = useTranslation();
  const id = useId();
  const isPilot = useIsPilot();
  const missionId = useLiveStore((s) => s.mission?.id ?? null);
  const [text, setText] = useState("");
  const tooLong = text.length > CHAT_MAX_CHARS;
  const canSend = isPilot && missionId !== null && text.trim().length > 0 && !tooLong;

  async function send(): Promise<void> {
    if (!canSend || missionId === null) return;
    const clientMsgId = uuidv7();
    const body = text;
    setText("");
    const chat = useChatStore.getState();
    chat.addLocal({
      key: clientMsgId,
      clientMsgId,
      role: "operator",
      kind: "text",
      text: body,
      createdAt: new Date().toISOString(),
      status: "sending",
    });
    try {
      const saved = await api<ChatMessage>(paths.missionChat(missionId), {
        method: "POST",
        body: { client_msg_id: clientMsgId, text: body, annotation_ids: [] },
      });
      chat.receive({ ...saved, refs: { ...saved.refs, client_msg_id: clientMsgId } });
    } catch {
      chat.markFailed(clientMsgId);
    }
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
      <label htmlFor={id} className="sr-only">
        {t("chat.inputLabel")}
      </label>
      <textarea
        id={id}
        className="input"
        rows={2}
        value={text}
        disabled={!isPilot}
        placeholder={t("chat.placeholder")}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
      />
      <button type="submit" className="btn btn-primary" disabled={!canSend}>
        {t("chat.send")}
      </button>
      {text.length >= COUNTER_FROM ? (
        <span className={tooLong ? "error" : "muted"}>
          {t("chat.counter", { n: text.length, max: CHAT_MAX_CHARS })}
        </span>
      ) : null}
      {hint ? <p className="muted composer-hint">{hint}</p> : null}
    </form>
  );
}

export function ChatView({ vehicleId }: ViewProps) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  return (
    <div className="view-body chat" id="chat">
      <MessageList />
      <ResumeRow vehicleId={vehicleId} />
      {isPilot ? (
        <fieldset className="orders" aria-describedby="orders-note">
          <legend className="sr-only">{t("orders.legend")}</legend>
          {ORDER_BUTTONS.map((k) => (
            <button
              key={k}
              type="button"
              className="btn btn-small"
              disabled
              data-testid="order-button"
            >
              {t(k)}
            </button>
          ))}
          <span id="orders-note" className="muted">
            {t("orders.notYet")}
          </span>
        </fieldset>
      ) : null}
      <Composer />
    </div>
  );
}
