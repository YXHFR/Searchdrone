import { Hand } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useIsPilot, useSessionStore } from "../../state/sessionStore";
import type { ViewProps } from "../registry";
import { Stick } from "./Stick";
import { type Axes, axesFromSticks, NEUTRAL } from "./sticks";
import { useGamepadName } from "./useGamepad";

/**
 * Manual control: "Take over" pauses the AI and the vehicle holds in LOITER; sticks on screen
 * or a game controller fly it; "Hand back" returns control. In this version the frame is in
 * place but nothing is sent: the button stays off until the ground server supports it.
 */
export default function ManualView(_: ViewProps) {
  const { t } = useTranslation();
  const isPilot = useIsPilot();
  const enabled = useSessionStore((s) => s.config?.features.manual_control === true);
  const gamepad = useGamepadName();
  const [left, setLeft] = useState({ x: 0, y: 0 });
  const [right, setRight] = useState({ x: 0, y: 0 });
  // Sticks only move once control is taken over; that path is not wired yet.
  const active = false;
  const axes: Axes = active ? axesFromSticks(left, right) : NEUTRAL;
  const fmt = (x: number): string => x.toFixed(2);

  return (
    <div className="view-body manual">
      <div className="manual-head">
        <button
          type="button"
          className="btn btn-primary"
          data-testid="takeover"
          disabled={!isPilot || !enabled || !active}
          aria-describedby="takeover-note"
        >
          <Hand aria-hidden="true" size={18} />
          {t("manual.takeover")}
        </button>
        <p id="takeover-note" className="muted">
          {isPilot ? t("manual.notYet") : t("manual.pilotOnly")}
        </p>
      </div>
      <p className="muted">{t("manual.explain")}</p>
      <div className="sticks">
        <Stick label={t("manual.leftStick")} disabled={!active} onChange={setLeft} />
        <Stick label={t("manual.rightStick")} disabled={!active} onChange={setRight} />
      </div>
      <p className="muted" data-testid="manual-axes">
        {t("manual.axes", {
          roll: fmt(axes.roll),
          pitch: fmt(axes.pitch),
          yaw: fmt(axes.yaw),
          throttle: fmt(axes.throttle),
        })}
      </p>
      <p className="muted">
        {gamepad ? t("manual.gamepad", { name: gamepad }) : t("manual.noGamepad")}
      </p>
      <p className="muted">{t("manual.rcFirst")}</p>
    </div>
  );
}
