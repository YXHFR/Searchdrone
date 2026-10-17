import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, HttpError } from "../../api/http";
import { paths } from "../../api/paths";
import type { CommandResult } from "../../api/types";
import { uuidv7 } from "../../lib/uuidv7";
import { useLiveStore } from "../../state/liveStore";
import { activeCharter, useMissionStore } from "../../state/missionStore";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import { useUiStore } from "../../state/uiStore";

interface Order {
  key: string;
  label: string;
  skill: string;
  params: Record<string, unknown>;
}

const MAX_PINS = 3;

/**
 * Order buttons as a fallback to the chat: the same skills the AI uses, sent as the pilot's
 * own orders. They go through the same safety gate; a refusal is shown, never bypassed.
 */
export function OrderButtons() {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const enabled = useSessionStore((s) => s.config?.features.order_buttons !== false);
  const mission = useLiveStore((s) => s.mission);
  const tel = useLiveStore((s) => s.telemetry);
  const activeId = mission?.active_charter_id ?? null;
  const charter = useMissionStore((s) => activeCharter(s, activeId));
  const annotations = useMissionStore((s) => s.annotations);
  const sectors = useMissionStore((s) => s.sectors);
  const [busy, setBusy] = useState<string | null>(null);
  if (!isPilot || !enabled || !mission || !charter) return null;

  const alt = charter.body.default_alt_agl_m ?? charter.body.alt_min_agl_m ?? 30;
  const airborne = tel?.armed === true && tel.landed_state === "IN_AIR";
  const orders: Order[] = [];
  if (!airborne) {
    orders.push({
      key: "takeoff",
      label: t("orders.takeoff"),
      skill: "takeoff",
      params: { alt_m: alt },
    });
  } else {
    const pins = Object.values(annotations)
      .filter((a) => a.kind === "pin")
      .sort((a, b) => a.created_at.localeCompare(b.created_at))
      .slice(-MAX_PINS);
    for (const p of pins) {
      const c = (p.geometry as { coordinates?: number[] }).coordinates;
      if (!c || typeof c[0] !== "number" || typeof c[1] !== "number") continue;
      orders.push({
        key: `goto-${p.id}`,
        label: t("orders.gotoPin", { pin: p.label }),
        skill: "goto",
        params: { lat: c[1], lon: c[0], alt_m: Math.round(tel?.alt_rel_home_m ?? alt) },
      });
    }
    const next = sectors.find((s) => s.properties.status !== "done" && s.properties.progress < 100);
    if (next) {
      orders.push({
        key: `survey-${next.properties.sector_id}`,
        label: t("orders.survey", { sector: next.properties.name }),
        skill: "survey_sector",
        params: { sector_id: next.properties.sector_id },
      });
    }
    orders.push({ key: "hold", label: t("orders.hold"), skill: "hold", params: {} });
  }

  async function send(o: Order): Promise<void> {
    if (!mission) return;
    setBusy(o.key);
    try {
      const r = await api<CommandResult>(paths.missionCommands(mission.id), {
        method: "POST",
        body: { client_command_id: uuidv7(), skill: o.skill, params: o.params },
      });
      useMissionStore.getState().upsertCommand(r.command);
      if (r.gate.decision === "REJECT")
        useUiStore.getState().toast(t("orders.rejected", { reason: r.gate.message_sv }));
      else if (r.gate.decision === "PENDING_APPROVAL")
        useUiStore.getState().toast(t("orders.needsApproval"));
    } catch (e) {
      useUiStore.getState().toast(e instanceof HttpError ? e.message : t("orders.failed"));
    } finally {
      setBusy(null);
    }
  }

  return (
    <fieldset className="orders">
      <legend className="sr-only">{t("orders.legend")}</legend>
      {orders.map((o) => (
        <button
          key={o.key}
          type="button"
          className="btn btn-small btn-outline"
          data-testid={`order-${o.skill}`}
          disabled={busy !== null}
          onClick={() => void send(o)}
        >
          {o.label}
        </button>
      ))}
    </fieldset>
  );
}
