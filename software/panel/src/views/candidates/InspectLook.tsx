// The open closer look on the candidate in review: the strip with the step, the height and
// the floor; the drone's "Gå lägre?" (only the pilot answers); and, once the steps are over,
// the countdown to the decision. STOP and FLY HOME stay in the safety bar the whole time.
//
// Screen readers: the height on the strip changes twice a second and the countdowns every
// second, so none of them is a live region. The strip tells the step and the phase when
// they change; the question and the end of the steps come as toasts (api/live.ts).
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { Inspection } from "../../api/types";
import { mmss, useNow } from "../../lib/useNow";
import { useInspectStore } from "../../state/inspectStore";
import { useIsPilot } from "../../state/sessionStore";
import { leftMs, metres, stripFloor } from "./inspect";

export function InspectStrip({ look }: { look: Inspection }) {
  const { t } = useTranslation();
  const p = useInspectStore((s) => s.progress[look.command_id] ?? null);
  const title = t("inspect.strip.title", { seq: look.seq });
  const phase = look.phase === "flying" ? (p?.phase ?? "approach") : look.phase;
  const where =
    phase === "step" && p
      ? t("inspect.strip.step", { n: p.step })
      : t(`inspect.strip.phase.${phase}`);
  const parts = [title, where];
  const agl = metres(p?.above_ground_m);
  if (agl !== null) parts.push(t("inspect.strip.agl", { m: agl }));
  const floor = stripFloor(p);
  if (floor !== null)
    parts.push(
      t("inspect.strip.floor", {
        m: floor,
        why: t(`inspect.floorReason.${p?.floor_reason ?? "map"}`),
      }),
    );
  if (p) parts.push(t(p.laser_confirmed ? "inspect.strip.laserOn" : "inspect.strip.laserOff"));
  return (
    <p className="inspect-strip" data-testid="inspect-strip" data-phase={look.phase}>
      {parts.join(" · ")}
      {look.overflight ? (
        <span className="inspect-over">{t(`inspect.strip.over.${look.operation_mode}`)}</span>
      ) : null}
      {/* Told only when the step or the phase changes, not with every new height. */}
      <span className="sr-only" role="status" data-testid="inspect-strip-said">
        {`${title} · ${where}`}
      </span>
    </p>
  );
}

/** "Gå lägre, ned till X m över mark?" with the drone's 60 s. Only the pilot answers: Ja
 * takes the drone down to the lowest floor, Nej tells it at once to stay at its height
 * (``/stay``; it never goes lower for it). No answer leaves it there when the time runs out. */
export function LowerOffer({ look }: { look: Inspection }) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const now = useNow(1_000);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const offer = look.lower_offer;
  if (look.phase !== "lower_offer" || !offer) return null;
  const left = leftMs(look.lower_offer_until, now);
  const to = metres(offer.lowest_above_ground_m) ?? 0;
  const answer = async (yes: boolean): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      const url = yes
        ? paths.inspectionLower(look.command_id)
        : paths.inspectionStay(look.command_id);
      const updated = await api<Inspection>(url, { method: "POST" });
      useInspectStore.getState().upsert(updated);
    } catch (e) {
      setError(e instanceof HttpError ? e.message : t("inspect.failed"));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="inspect-box inspect-lower" data-testid="inspect-lower">
      <h3>{t("inspect.lower.question", { m: to })}</h3>
      <p className="muted">
        {t("inspect.lower.now", { m: metres(offer.now_above_ground_m) ?? 0 })}
        {left !== null ? ` · ${t("inspect.lower.left", { left: mmss(left) })}` : ""}
      </p>
      {!isPilot ? <p className="muted">{t("inspect.lower.pilotOnly")}</p> : null}
      {isPilot ? (
        <div className="cand-buttons">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            data-testid="inspect-lower-yes"
            onClick={() => void answer(true)}
          >
            {t("inspect.lower.yes")}
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            data-testid="inspect-lower-no"
            onClick={() => void answer(false)}
          >
            {t("inspect.lower.no")}
          </button>
        </div>
      ) : null}
      {error ? (
        <p className="warn" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}

/** The steps are over: how they ended and the time left for the decision. */
export function AwaitBox({ look }: { look: Inspection }) {
  const { t } = useTranslation();
  const now = useNow(1_000);
  if (look.phase !== "await") return null;
  const left = leftMs(look.await_until, now);
  const done = look.done;
  return (
    <section className="inspect-box inspect-await" data-testid="inspect-await">
      <h3 data-testid="inspect-countdown">
        {t("inspect.await.title", { left: mmss(left ?? 0), seq: look.seq })}
      </h3>
      {done ? (
        <p className="muted" data-testid="inspect-end">
          {t(`inspect.end.${done.end}`, {
            m: metres(done.lowest_above_ground_m) ?? "–",
          })}
          {" · "}
          {t("inspect.await.steps", { n: done.steps })}
        </p>
      ) : null}
    </section>
  );
}
