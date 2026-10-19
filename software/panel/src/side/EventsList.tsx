import { useTranslation } from "react-i18next";
import { useUiStore } from "../state/uiStore";

const time = new Intl.DateTimeFormat("sv-SE", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

/** Händelser: what the system did and reported, newest first, beside the AI's conversation. */
export function EventsList() {
  const { t } = useTranslation();
  const log = useUiStore((s) => s.eventLog);
  if (log.length === 0)
    return (
      <p className="side-empty" data-testid="events">
        {t("status.log.empty")}
      </p>
    );
  return (
    <ol className="events" data-testid="events">
      {log.map((e) => (
        <li key={`${e.t}-${e.message}`} data-level={e.level}>
          <time className="num">{time.format(new Date(e.t))}</time>
          <span>{e.message}</span>
        </li>
      ))}
    </ol>
  );
}
